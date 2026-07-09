import dataclasses
import datetime
import functools
import math
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm

from reconstruction.data.capturestudio import MultiSessionDataset
from reconstruction.merging.blendpcr import blend_point_cloud
from reconstruction.merging.depth_fusion import fuse_depth_maps
from reconstruction.primitive.pcd import RGBDImage
from reconstruction.primitive.splat import GSImage
from reconstruction.vis.teaser.base import (
    CapturestudioVirtualBackgroundFloorWallEstimator,
    CapturestudioVirtualBackground,
    CapturestudioVirtualCameras,
    CapturestudioVirtualDynamicForeground,
    CapturestudioVirtualScene,
    TeaserGenerator,
    TeaserGeneratorRenderConfig
)
from utils.misc import log


# ---------------------------------------------------------------------------
# Pipeline Classes for Open3D
# ---------------------------------------------------------------------------
class CapturestudioVirtualBackgroundFloorWallEstimatorOpen3D(CapturestudioVirtualBackgroundFloorWallEstimator):
    pass


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualBackgroundOpen3D(CapturestudioVirtualBackground):

    def to_open3d_scene(self, scene: o3d.visualization.rendering.Open3DScene) -> List[str]:
        added_names = []

        # Global SE(3) transform
        Rg = np.asarray(self.R_scene, dtype=np.float64) if self.R_scene is not None else np.eye(3, dtype=np.float64)
        tg = np.asarray(self.t_scene, dtype=np.float64).reshape(-1) if self.t_scene is not None else np.zeros(3, dtype=np.float64)

        def _make_quad_geometry(corners, tex_path, name):
            if corners is None or len(corners) < 3: return None
            # Prepare vertices
            c = (Rg @ corners[:4].T).T + tg
            verts = np.array([c[0], c[1], c[2], c[3]], dtype=np.float64)
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

            # Open3D expects UVs per triangle vertex (6 total for a quad)
            uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.float64)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

            # Load Texture using OpenCV to guarantee correct color space, then to Open3D
            img = cv2.imread(str(tex_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_o3d = o3d.geometry.Image(img)
            # # Convert to float, apply inverse gamma, convert back to uint8 (or pass as float32)
            # img_linear = np.power(img.astype(np.float32) / 255.0, 2.2)
            # img_linear = (img_linear * 255.0).astype(np.uint8)
            # img_o3d = o3d.geometry.Image(img_linear)

            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.albedo_img = img_o3d

            scene.add_geometry(name, mesh, mat)
            return name

        if self.bg_type == 'floor_wall':
            rf = _make_quad_geometry(self.floor_corners, self.floor_texture_path, 'bg:floor')
            if rf: added_names.append(rf)

            rw = _make_quad_geometry(self.wall_corners, self.wall_texture_path, 'bg:wall')
            if rw: added_names.append(rw)

        elif self.bg_type in ['ply:pcd', 'ply:mesh']:
            if isinstance(self.static_ply, trimesh.Geometry):
                geometries = self.static_ply.geometry.values() if isinstance(self.static_ply, trimesh.Scene) else [self.static_ply]
                for i, geom in enumerate(geometries):
                    if isinstance(geom, trimesh.PointCloud):
                        pts = (Rg @ geom.vertices.T).T + tg
                        cols = geom.colors[:, :3] / 255.0 if hasattr(geom, 'colors') and geom.colors is not None else np.full((len(pts), 3), 0.5)

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pts)
                        pcd.colors = o3d.utility.Vector3dVector(cols)

                        mat = o3d.visualization.rendering.MaterialRecord()
                        mat.shader = "defaultUnlit"
                        mat.point_size = 3.0

                        name = f'bg:static_pcd_{i}'
                        scene.add_geometry(name, pcd, mat)
                        added_names.append(name)
            else:
                log("Mesh background rendering not fully implemented in this script.", "warning")

        elif self.bg_type == 'ply:splat':
            raise NotImplementedError("Gaussian Splatting is not supported in Open3D renderer yet.")

        return added_names

    @classmethod
    def from_capturestudio_dataset(cls, *args, **kwargs) -> 'CapturestudioVirtualBackgroundOpen3D':
        bg = super().from_capturestudio_dataset(*args, **kwargs)
        bg.__class__ = cls
        return bg

    @classmethod
    def from_object(cls, *args, **kwargs) -> 'CapturestudioVirtualBackgroundOpen3D':
        bg = super().from_object(*args, **kwargs)
        bg.__class__ = cls
        return bg


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualCamerasOpen3D(CapturestudioVirtualCameras):

    def tick(self, scene: o3d.visualization.rendering.Open3DScene) -> Optional[Tuple[int, int]]:
        super_out = super().tick()
        if super_out is None: return None
        t_prev, t_next = super_out

        # Handle highlight toggles
        if self.camera_show_virtual_frusta:
            prev_key = f"virt:{t_prev}"
            next_key = f"virt:{t_next}"

            # Open3D allows material updates directly on the geometry
            dim_mat = o3d.visualization.rendering.MaterialRecord()
            dim_mat.shader = "unlitLine"
            dim_mat.line_width = 1.0
            dim_mat.base_color = [0.5, 0.5, 0.5, 1.0]

            act_mat = o3d.visualization.rendering.MaterialRecord()
            act_mat.shader = "unlitLine"
            act_mat.line_width = 3.0
            act_mat.base_color = [1.0, 0.0, 0.0, 1.0]

            if scene.has_geometry(prev_key):
                scene.modify_geometry_material(prev_key, dim_mat)
            if scene.has_geometry(next_key):
                scene.modify_geometry_material(next_key, act_mat)

        return super_out

    def to_open3d_scene(self, scene: o3d.visualization.rendering.Open3DScene, R_scene: Optional[np.ndarray] = None, t_scene: Optional[np.ndarray] = None) -> List[str]:
        added_names = []

        def _build_frustum(K, c2w, name, is_active):
            c2w_v = np.eye(4, dtype=np.float64)
            c2w_v[:3, :3] = (R_scene @ c2w[:3, :3]) if R_scene is not None else c2w[:3, :3]
            c2w_v[:3, 3] = ((R_scene @ c2w[:3, 3]) + t_scene) if R_scene is not None else c2w[:3, 3]

            scale = self.frustum_scale
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            w, h = self.image_size_hw[1], self.image_size_hw[0]

            # 5 points: origin, and 4 corners
            pts_cam = np.array([
                [0, 0, 0],
                [(0 - cx) * scale / fx, (0 - cy) * scale / fy, scale],
                [(w - cx) * scale / fx, (0 - cy) * scale / fy, scale],
                [(w - cx) * scale / fx, (h - cy) * scale / fy, scale],
                [(0 - cx) * scale / fx, (h - cy) * scale / fy, scale],
            ], dtype=np.float64)

            pts_world = (c2w_v[:3, :3] @ pts_cam.T).T + c2w_v[:3, 3]

            lines_idx = np.array([
                [0, 1], [0, 2], [0, 3], [0, 4],  # Rays
                [1, 2], [2, 3], [3, 4], [4, 1]  # Image plane
            ], dtype=np.int32)

            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts_world)
            ls.lines = o3d.utility.Vector2iVector(lines_idx)

            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "unlitLine"
            if is_active:
                mat.line_width = 3.0
                mat.base_color = [1.0, 0.0, 0.0, 1.0]
            else:
                mat.line_width = 1.0
                mat.base_color = [0.5, 0.5, 0.5, 1.0]

            scene.add_geometry(name, ls, mat)
            return name

        if self.camera_show_virtual_frusta and self.virtual_extrinsics_c2ws is not None:
            self._compute_lookat_if_needed(R_scene, t_scene)
            self._roll_align_first_to_gt(R_scene, t_scene)
            for i in range(len(self.virtual_extrinsics_c2ws)):
                is_active = (i == self._t_current)
                c2w_prep = self._prepare_virtual_c2w(i, R_scene, t_scene)
                name = f"virt:{i}"
                _build_frustum(self.virtual_intrinsics[i], c2w_prep, name, is_active)
                added_names.append(name)

        self._R_scene = R_scene
        self._t_scene = t_scene
        return added_names

    @classmethod
    def from_capturestudio_dataset(cls, *args, **kwargs) -> 'CapturestudioVirtualCamerasOpen3D':
        c = super().from_capturestudio_dataset(*args, **kwargs)
        c.__class__ = cls
        return c


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualDynamicForegroundOpen3D(CapturestudioVirtualDynamicForeground):

    @functools.cached_property
    def label(self) -> str:
        return f'fg_{self.index:02d}'

    def tick(self, new_active_cam_idx: Optional[int] = None, target_w2c: Optional[np.nditer] = None, scene: Optional[o3d.visualization.rendering.Open3DScene] = None) -> None:
        super().tick(new_active_cam_idx=new_active_cam_idx)
        if self._last_vis_args is not None and len(self._last_vis_args) == 3:
            self.to_open3d_scene(*self._last_vis_args, target_w2c=target_w2c)
        else:
            self.to_open3d_scene(scene=scene, target_w2c=target_w2c)

    def to_open3d_scene(self, scene: o3d.visualization.rendering.Open3DScene, R_scene: Optional[np.ndarray] = None, t_scene: Optional[np.ndarray] = None, target_w2c: Optional[np.nditer] = None) -> Optional[List[str]]:
        self._last_vis_args = (scene, R_scene, t_scene)
        image = next(self.image_generator)

        if isinstance(image, list):
            if self._blending_strategy == 'swap':
                image = image[self._active_cam_idx]
            elif self._blending_strategy == 'fuse':
                image = fuse_depth_maps(image)[self._active_cam_idx]
            elif self._blending_strategy.startswith('blend'):
                if target_w2c is None:
                    target_w2c = image[self._active_cam_idx].extrinsic_w2c
                closest_idx = list({max(0, self._active_cam_idx - 1), self._active_cam_idx})
                image = blend_point_cloud([image[i] for i in closest_idx], target_w2c=target_w2c, voxel_size_m=0.006, angle_power=4.0, refine_registration='deformation_pyramid')

        if isinstance(image, GSImage):
            raise NotImplementedError("Gaussian Splatting is not supported in Open3D renderer yet.")

        if isinstance(image, RGBDImage):
            image = image.unproject().open3d

        if isinstance(image, o3d.geometry.PointCloud):
            pts = np.asarray(image.points, dtype=np.float64)
            colors = np.asarray(image.colors, dtype=np.float64)

            if R_scene is not None:
                pts = (R_scene @ pts.T).T
            if t_scene is not None:
                pts += t_scene

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            # Convert sRGB to Linear space to prevent Open3D from double-brightening
            pcd.colors = o3d.utility.Vector3dVector(np.power(colors, 2.2))

            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 3.0

            name = f'{self.label}/pcd'

            # Fast swap via Scene graph
            if scene.has_geometry(name):
                scene.remove_geometry(name)
            scene.add_geometry(name, pcd, mat)

            return [name]

    @classmethod
    def from_capturestudio_dataset(cls, *args, **kwargs) -> 'CapturestudioVirtualDynamicForegroundOpen3D':
        fg = super().from_capturestudio_dataset(*args, **kwargs)
        fg.__class__ = cls
        return fg


class CaptureStudioVirtualSceneOpen3D(CapturestudioVirtualScene):
    def __init__(self, renderer: o3d.visualization.rendering.OffscreenRenderer, *args, **kwargs):
        self.renderer = renderer
        self._all_geometry_names = []
        super().__init__(*args, **kwargs)

    def capture(self, target_size_hw: Tuple[int, int], show_background: bool = True) -> np.ndarray:
        H, W = target_size_hw

        # Setup background visibility
        for name in self._all_geometry_names:
            if name.startswith('bg:'):
                self.renderer.scene.show_geometry(name, show_background)

        # Get Camera Math
        cam = self._cameras.current_camera
        K = np.asarray(cam["intrinsics"], np.float64)
        c2w = np.asarray(cam["extrinsic_c2w"], np.float64)

        fy = float(K[1, 1])
        fov_y_deg = math.degrees(2.0 * math.atan(H / (2.0 * fy)))

        # Open3D Camera follows OpenGL convention internally, but setup_camera accepts standard math
        eye = c2w[:3, 3]
        forward = c2w[:3, 2]  # OpenCV +Z is forward
        up = -c2w[:3, 1]  # OpenCV +Y is down -> -Y is up

        self.renderer.setup_camera(fov_y_deg, eye + forward, eye, up)

        # Render pass
        img = self.renderer.render_to_image()
        image_np = np.asarray(img)

        return image_np

    def on_scene_updated(self, what_updated: str, set_not_unset: bool = True) -> None:
        if set_not_unset:
            kwargs = {}
            if what_updated != 'background':
                kwargs = dict(R_scene=self._R_scene, t_scene=self._t_scene)
            if what_updated == 'foregrounds' and self._cameras is not None:
                kwargs['target_w2c'] = np.linalg.inv(self._cameras.current_camera['extrinsic_c2w'])

            attr = getattr(self, f'_{what_updated}')
            if not isinstance(attr, list):
                attr = [attr]

            for i, attr_i in enumerate(attr):
                added_names = attr_i.to_open3d_scene(self.renderer.scene, **kwargs)
                if added_names:
                    self._all_geometry_names.extend(added_names)
        else:
            # Handle teardown
            to_remove = [n for n in self._all_geometry_names if n.startswith(f'{what_updated}:')]
            for n in to_remove:
                if self.renderer.scene.has_geometry(n):
                    self.renderer.scene.remove_geometry(n)
            self._all_geometry_names = [n for n in self._all_geometry_names if n not in to_remove]

    def tick(self):
        for _asset in ['background', 'cameras', 'foregrounds']:
            attr = getattr(self, f'_{_asset}')
            if attr is None or not hasattr((attr if not isinstance(attr, list) else attr[0]), 'tick'):
                continue
            if not isinstance(attr, list):
                attr = [attr]
            for i, attr_i in enumerate(attr):
                kwargs = dict(scene=self.renderer.scene)
                if _asset == 'foregrounds' and self._cameras is not None:
                    kwargs['new_active_cam_idx'] = self._cameras.gt_cam_index
                    kwargs['target_w2c'] = np.linalg.inv(self._cameras.current_camera['extrinsic_c2w'])
                attr_i.tick(**kwargs)

    @classmethod
    def from_capturestudio_session(cls, renderer: o3d.visualization.rendering.OffscreenRenderer, dataset_raw: MultiSessionDataset, **kwargs) -> 'CaptureStudioVirtualSceneOpen3D':
        dataset_vis = kwargs.pop('dataset_vis', dataset_raw)
        wall_overshoot_m = kwargs.pop('wall_overshoot_m', 1.3)
        use_gs = kwargs.pop('use_gs', False)
        t_start = kwargs.pop('t_start', 0)
        t_total = kwargs.pop('t_total', -1)
        camera_orbit_type = kwargs.pop('camera_orbit_type', 'audience')

        if not isinstance(dataset_vis, list): dataset_vis = [dataset_vis]
        if not isinstance(t_start, list): t_start = [t_start] * len(dataset_vis)
        if not isinstance(t_total, list): t_total = [t_total] * len(t_start)

        floor_wall_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith('floor') or k.startswith('wall')}
        bg_kwargs = {k[3:]: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith('bg_')}
        fg_kwargs = {k[3:]: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith('fg_')}

        if 'obj_path' in bg_kwargs:
            scene_background = CapturestudioVirtualBackgroundOpen3D.from_object(**bg_kwargs)
        else:
            scene_background = CapturestudioVirtualBackgroundOpen3D.from_capturestudio_dataset(dataset_raw, wall_overshoot_m=wall_overshoot_m, **floor_wall_kwargs)

        scene_cameras = CapturestudioVirtualCamerasOpen3D.from_capturestudio_dataset(
            dataset_raw, background=scene_background, camera_orbit_type=camera_orbit_type, t_total=max(t_total),
            **{k: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith('camera')}
        )

        scene_foregrounds = [
            CapturestudioVirtualDynamicForegroundOpen3D.from_capturestudio_dataset(
                dataset_vis_i, use_gs=use_gs, t_start=t_start_i, t_total=t_total_i, index=i,
                active_cam_idx=scene_cameras.gt_cam_index, **fg_kwargs
            )
            for i, (dataset_vis_i, t_start_i, t_total_i) in enumerate(zip(dataset_vis, t_start, t_total))
        ]

        scene = cls(renderer, **kwargs)
        scene.set_background(background=scene_background)
        scene.set_foregrounds(foregrounds=scene_foregrounds)
        scene.set_cameras(cameras=scene_cameras)
        return scene


class TeaserGeneratorOpen3D(TeaserGenerator):
    def __init__(self, *args, **kwargs):
        self._show_background = kwargs.pop('show_background', True)
        # Delay renderer init until config is loaded
        self.renderer = None
        super().__init__(*args, **kwargs)

    def _init_scene(self):
        if self.t_total == -1:
            self.t_total = min(len(_) for _ in self.datasets_vis)

        H, W = self.render_config.image_size_hw

        # Initialize Offscreen Renderer
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        self.renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

        self._scene = CaptureStudioVirtualSceneOpen3D.from_capturestudio_session(
            renderer=self.renderer,
            dataset_raw=self.dataset_raw,
            dataset_vis=self.datasets_vis,
            t_start=self.t_start,
            t_total=self.t_total + 1,
            use_gs=self.render_config.use_gs,
            camera_orbit_type=self.render_config.camera_orbit_type,
            camera_traverse_velocity=self.render_config.camera_traverse_velocity,
            camera_orbit_offset_m=self.render_config.camera_orbit_offset_m,
            camera_show_gt_frusta=self.render_config.camera_show_gt_frusta,
            camera_show_virtual_frusta=self.render_config.camera_show_virtual_frusta,
            floor_depth_scale=self.render_config.floor_depth_scale,
            wall_overshoot_m=self.render_config.wall_overshoot_m,
            wall_pad_width_m=self.render_config.wall_pad_width_m,
            **self._scene_kwargs
        )

    def _rendering_loop(self) -> Path:
        video_writer_ = None
        for _ in tqdm(range(self.t_total), desc=f'Generating video file "{self.out_video_path}"'):
            rendered_img_ = self._scene.capture(self.render_config.image_size_hw, show_background=self._show_background)

            if self.debug:
                out_file = Path(self.out_video_path).name.split('.')[0] + f'_t0{str(datetime.timedelta(milliseconds=self.t_start * 1 / 30 * 1000))[:-3].replace(":", "-").replace(".", "-")}.png'
                cv2.imwrite(out_file, cv2.cvtColor(rendered_img_, cv2.COLOR_RGB2BGR))
                print(f'done: {out_file}')
                exit(-1)

            if video_writer_ is None:
                video_writer_ = cv2.VideoWriter(
                    self.out_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    30,
                    (rendered_img_.shape[1], rendered_img_.shape[0])
                )

            # Open3D render output is RGB, convert to BGR for OpenCV
            video_writer_.write(cv2.cvtColor(rendered_img_, cv2.COLOR_RGB2BGR))
            self._scene.tick()

        if video_writer_ is not None:
            video_writer_.release()
            log(f'[{self.__class__.__name__}::_rendering_loop] Video file released: "{self.out_video_path}"')

        return self.out_video_path

    def run(self) -> Path:
        return self._rendering_loop()


if __name__ == '__main__':
    # @formatter:off
    DATA = [
        ('Cagliari_1_Perf_7',    'Cagliari_1_Calib_6',    0,    list(range(1, 8))),
    ]
    # @formatter:on
    DEBUG = False

    import time as time_
    import gc as gc_
    import torch

    for SESSION_PERF, SESSION_CALIB, T_START, CAM_IDX in DATA:
        video_paths_ = {}
        for use_gs_ in [False]:
            visualizer_ = TeaserGeneratorOpen3D(
                session_perf=SESSION_PERF if isinstance(SESSION_PERF, list) else SESSION_PERF.split('|'),
                session_calib=SESSION_CALIB,
                calib_method='MultiCamCalib',
                depth_source='bilateral_temporal',
                cam_idx_perf=CAM_IDX,
                cam_idx_raw=CAM_IDX,
                render_config=TeaserGeneratorRenderConfig.for_apr_may_2025(
                    use_gs=use_gs_,
                    image_size_hw=(1080, 1920),
                    camera_traverse_velocity=0.4,
                    camera_orbit_offset_m=0.6,
                ),
                t_start=T_START,
                t_total=1 if DEBUG else -1,  # reduced for quick tests
                show_background=True,
                camera_orbit_start_idx=T_START,
                fg_blending_strategy='swap'
            )
            video_path_ = visualizer_.run()
            video_paths_[f"{'gs' if use_gs_ else 'pcd'}_path"] = video_path_

        # Cleanup
        del visualizer_
        gc_.collect()
        torch.cuda.empty_cache()
        time_.sleep(1.0)
