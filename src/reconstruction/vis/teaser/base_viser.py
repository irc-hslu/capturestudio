import dataclasses
import datetime
import functools
import time
from pathlib import Path
from typing import Tuple, Optional, Literal, List, Union, Dict, Type

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
import viser
from kornia.geometry import rotation_matrix_to_quaternion
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
from utils.misc import PathUtils, log


class CapturestudioVirtualBackgroundFloorWallEstimatorViser(CapturestudioVirtualBackgroundFloorWallEstimator):
    def __call__(self, *args, vis_viser_server: Optional['viser.ViserServer'] = None, **kwargs) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray]:
        floor_normal, floor_offset, floor_corners, wall_normal, wall_offset, wall_corners = super().__call__(*args, **kwargs)

        if vis_viser_server is not None:
            self.__class__.visualize_floor_and_wall_3d_viser(
                server=vis_viser_server,
                views=kwargs['views'] if 'views' in kwargs else args[0],
                floor_corners=floor_corners,
                wall_corners=wall_corners,
                floor_image=PathUtils.resources_path() / 'backdrops' / f'floor_dark.jpg',
                wall_image=PathUtils.resources_path() / 'backdrops' / f'curtain_dark_hslu.png',
            )

        return floor_normal, floor_offset, floor_corners, wall_normal, wall_offset, wall_corners

    @classmethod
    def visualize_floor_and_wall_3d_viser(
            cls,
            server: 'viser.ViserServer',
            views: List[RGBDImage],
            floor_corners: Optional[np.ndarray],
            wall_corners: Optional[np.ndarray],
            floor_image: Optional[Union[np.ndarray, Path, str]] = None,
            wall_image: Optional[Union[np.ndarray, Path, str]] = None,
            ref_view_index: Optional[int] = None,
            point_size: float = 0.0035,
            pc_name: str = '/pcd/ref',
            floor_name: str = '/bg/floor',
            wall_name: str = '/bg/wall'
    ) -> None:
        # ---- helpers ----
        _normalize = lambda v: v / (np.linalg.norm(v) + 1e-12)

        def _rx_pi():
            return np.diag([1.0, -1.0, -1.0])  # Open3D->Viser camera convention

        def _rot_a_to_b(a, b):
            a = _normalize(a)
            b = _normalize(b)
            v = np.cross(a, b)
            s = np.linalg.norm(v)
            c = float(np.dot(a, b))
            if s < 1e-10:
                if c > 0: return np.eye(3)
                # 180°: pick any axis ⟂ a
                axis = _normalize(np.cross(a, np.array([1.0, 0.0, 0.0])) if abs(a[0]) < 0.9 else np.cross(a, np.array([0.0, 1.0, 0.0])))
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]], dtype=np.float64)
                return np.eye(3) + 2.0 * (K @ K)  # Rodrigues with θ=π: I + 2K^2
            K = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]], dtype=np.float64)
            return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))

        # Ensure Viser scene uses its camera up (−Y) and shows axes
        server.scene.set_up_direction((0.0, -1.0, 0.0))
        server.scene.world_axes.visible = True
        up_vis = np.array([0.0, -1.0, 0.0], dtype=np.float64)

        # Need at least the floor to define alignment.
        if floor_corners is None or len(floor_corners) < 3:
            # Fallback: just add PCD if present, without alignment.
            if len(views) > 0:
                idx = int(ref_view_index if ref_view_index is not None else 0)
                ref = views[max(0, min(idx, len(views) - 1))]
                ref_pcd = ref.unproject().open3d
                pts = np.asarray(ref_pcd.points, dtype=np.float64)
                cols = np.asarray(ref_pcd.colors, dtype=np.float64)
                Rx = _rx_pi()
                pts_vis = (Rx @ pts.T).T
                server.scene.add_point_cloud(name=pc_name, points=pts_vis.astype(np.float32), colors=np.clip(cols, 0, 1).astype(np.float32), point_size=point_size)
            return

        # Ensure quads are 4-pt by synthesizing the 4th
        def _ensure_quad(c):
            c = np.asarray(c, dtype=np.float64)
            if c.shape[0] == 3:
                p0, p1, p2 = c
                p3 = p2 - (p1 - p0)
                return np.stack([p0, p1, p2, p3], 0)
            return c[:4].astype(np.float64)

        fc_w = _ensure_quad(floor_corners)
        wc_w = _ensure_quad(wall_corners) if wall_corners is not None and len(wall_corners) >= 3 else None

        # --- Build local bases in WORLD frame (Open3D convention) ---
        Rf_w, wf, hf, cf_w = cls.basis_from_corners(fc_w)

        # --- Global transform to Viser: Rx (Open3D->Viser camera) then align floor normal to up_vis ---
        Rx = _rx_pi()
        n_floor_w = _normalize(np.cross(fc_w[1] - fc_w[0], fc_w[3] - fc_w[0]))
        n_floor_vis = Rx @ n_floor_w
        R_align = _rot_a_to_b(n_floor_vis, up_vis)  # rotate so floor is perfectly horizontal
        R_full = R_align @ Rx

        # Find translation so floor lies on grid plane (y=0): use any floor corner after rotation.
        fc_vis = (R_full @ fc_w.T).T
        d = float(up_vis @ fc_vis[0])  # constant height after R_align
        t = -d * up_vis  # bring plane to y=0

        # --- Add floor image ---
        Rf_vis = R_full @ Rf_w
        cf_vis = R_full @ cf_w + t
        floor_wxyz = rotation_matrix_to_quaternion(torch.from_numpy(Rf_vis)).numpy()
        server.scene.add_image(
            name=floor_name,
            image=np.flipud(np.fliplr(cv2.imread(str(floor_image), cv2.IMREAD_COLOR_RGB))),
            render_width=wf,
            render_height=hf,
            wxyz=floor_wxyz,
            position=(float(cf_vis[0]), float(cf_vis[1]), float(cf_vis[2])),
        )

        # --- Add wall image ---
        if wc_w is not None:
            Rw_w, ww, hw, cw_w = cls.basis_from_corners(wc_w)
            Rw_vis = R_full @ Rw_w
            cw_vis = R_full @ cw_w + t
            wall_wxyz = rotation_matrix_to_quaternion(torch.from_numpy(Rw_vis)).numpy()
            server.scene.add_image(
                name=wall_name,
                image=np.flipud(np.fliplr(cv2.imread(str(wall_image), cv2.IMREAD_COLOR_RGB))),
                render_width=ww, render_height=hw,
                wxyz=wall_wxyz,
                position=(float(cw_vis[0]), float(cw_vis[1]), float(cw_vis[2])),
            )

        # --- Add one point cloud (transformed by the same global SE(3)) ---
        if len(views) > 0:
            idx = int(ref_view_index if ref_view_index is not None else next(
                (i for i, v in enumerate(views) if (v is not None and v.depth is not None)), 0
            ))
            idx = max(0, min(idx, len(views) - 1))
            ref = views[idx]
            ref_pcd = ref.unproject().open3d
            pts = np.asarray(ref_pcd.points, dtype=np.float64)
            cols = np.asarray(ref_pcd.colors, dtype=np.float64)
            pts_vis = (R_full @ pts.T).T + t
            server.scene.add_point_cloud(
                name=pc_name,
                points=pts_vis.astype(np.float32),
                colors=np.clip(cols, 0.0, 1.0).astype(np.float32),
                point_size=point_size,
            )


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualBackgroundViser(CapturestudioVirtualBackground):
    def _to_viser_scene_floor_wall(self, scene: viser.SceneApi) -> Dict[str, viser.SceneNodeHandle]:
        def _ensure_quad(c):
            if c is None:
                return None
            c = np.asarray(c, dtype=np.float64)
            if c.shape[0] < 3:
                return None
            if c.shape[0] == 3:
                p0, p1, p2 = c
                p3 = p2 - (p1 - p0)
                return np.stack([p0, p1, p2, p3], 0)
            return c[:4].astype(np.float64)

        # Make sure Viser grid uses +X right, +Z forward, -Y up.
        scene.set_up_direction((0.0, -1.0, 0.0))

        # Global SE(3) to Viser
        Rg = np.asarray(self.R_scene, dtype=np.float64) if self.R_scene is not None else np.eye(3, dtype=np.float64)
        tg = np.asarray(self.t_scene, dtype=np.float64).reshape(-1) if self.t_scene is not None else np.zeros(3, dtype=np.float64)

        added_handles = {}

        # ---------- Floor ----------
        fc = _ensure_quad(self.floor_corners)
        if fc is not None:
            Rw, w, h, ctr_w = CapturestudioVirtualBackgroundFloorWallEstimator.basis_from_corners(fc)
            Rv = Rg @ Rw
            ctr_v = Rg @ ctr_w + tg
            img_floor = np.flipud(np.fliplr(cv2.imread(str(self.floor_texture_path), cv2.IMREAD_COLOR_RGB)))
            hf = scene.add_image(
                name="/bg/floor",
                image=img_floor,
                render_width=float(w),
                render_height=float(h),
                wxyz=rotation_matrix_to_quaternion(torch.from_numpy(Rv)).numpy(),
                position=(float(ctr_v[0]), float(ctr_v[1]), float(ctr_v[2])),
            )
            added_handles['floor'] = hf

        # ---------- Wall ----------
        wc = _ensure_quad(self.wall_corners)
        if wc is not None:
            Rw, w, h, ctr_w = CapturestudioVirtualBackgroundFloorWallEstimator.basis_from_corners(wc)
            Rv = Rg @ Rw
            ctr_v = Rg @ ctr_w + tg
            img_wall = np.flipud(np.fliplr(cv2.imread(str(self.wall_texture_path), cv2.IMREAD_COLOR_RGB)))
            hw = scene.add_image(
                name="/bg/wall",
                image=img_wall,
                render_width=float(w),
                render_height=float(h),
                wxyz=rotation_matrix_to_quaternion(torch.from_numpy(Rv)).numpy(),
                position=(float(ctr_v[0]), float(ctr_v[1]), float(ctr_v[2])),
            )
            added_handles['wall'] = hw

        return added_handles

    def _to_viser_scene_ply(self, scene: viser.SceneApi, bg_kind: Literal['pcd', 'mesh', 'splat']) -> Dict[str, viser.SceneNodeHandle]:
        added_handles = {}

        # Global SE(3) transform (consistent with floor_wall visualization)
        global_R = np.asarray(self.R_scene, dtype=np.float64) if self.R_scene is not None else np.eye(3, dtype=np.float64)
        global_t = np.asarray(self.t_scene, dtype=np.float64).reshape(-1) if self.t_scene is not None else np.zeros(3, dtype=np.float64)
        global_tf = np.eye(4)
        global_tf[:3, :3] = global_R
        global_tf[:3, 3] = global_t

        # Process bg data
        if isinstance(self.static_ply, trimesh.Geometry):
            geometries = self.static_ply.geometry.values() if isinstance(self.static_ply, trimesh.Scene) else [self.static_ply]
            for i, geom in enumerate(geometries):
                geom.apply_transform(global_tf)
                if isinstance(geom, trimesh.Trimesh):
                    added_handles[f'static_ply_mesh_{i}'] = scene.add_mesh_trimesh(
                        name=f"/bg/mesh_{i}",
                        mesh=geom
                    )
                elif isinstance(geom, trimesh.PointCloud):
                    pts = geom.vertices if hasattr(geom, 'vertices') else geom.points
                    cols = geom.colors[:, :3] if hasattr(geom, 'colors') and geom.colors is not None else np.full((len(pts), 3), 150)
                    added_handles[f'static_ply_pcd_{i}'] = scene.add_point_cloud(
                        name=f"/bg/pcd_{i}",
                        points=pts.astype(np.float32),
                        colors=np.clip(cols / 255.0, 0.0, 1.0).astype(np.float32),
                        point_size=0.005
                    )
        elif bg_kind == 'splat' and isinstance(self.static_ply, dict):
            # transform
            from scipy.spatial.transform import Rotation as R
            centers = self.static_ply["centers"]
            centers_mean = centers.mean(axis=0)[None]
            centers = R.from_matrix(global_R).apply(centers - centers_mean) + global_t[None] + centers_mean
            covariances = self.static_ply['covariances']
            # TODO: rotate the covariances
            added_handles[f'static_ply_gs'] = scene.add_gaussian_splats(
                f"/bg/gs",
                centers=centers,
                rgbs=self.static_ply["rgbs"],
                opacities=self.static_ply["opacities"],
                covariances=covariances,
                # sh_coeffs=self.static_ply["sh_coeffs"] if self.static_ply['sh_coeffs'] and not np.all(self.static_ply['sh_coeffs'][3:] == 0) else None,
            )
        else:
            raise NotImplementedError('Static ply should either be a trimesh or a splat dict')

        # Debug: show annotations
        debug = True
        extra_handles = []
        if debug:
            floor_corners, wall_corners = self.floor_corners, self.wall_corners
            for i, (fc, wc) in enumerate(zip(floor_corners, wall_corners)):
                fc_h = scene.add_icosphere(
                    f"/bg/corners/floor_{i}",
                    radius=0.1,
                    color=(1.0, 0.0, 0.0),  # red
                    position=fc,
                )
                wc_h = scene.add_icosphere(
                    f"/bg/corners/wall_{i}",
                    radius=0.1,
                    color=(0.0, 0.0, 1.0),  # blue
                    position=wc,
                )
                extra_handles.extend([fc_h, wc_h])
            faces = np.asarray([[0, j, j + 1] for j in range(1, 3)], dtype=np.uint32)
            fp_h = scene.add_mesh_simple(
                f"/bg/planes/floor",
                vertices=floor_corners,
                faces=faces.copy(),
                color=(1.0, 0.0, 0.0),  # red
                opacity=0.98,
                side="double",
                cast_shadow=False,
                receive_shadow=False
            )
            wp_h = scene.add_mesh_simple(
                f"/bg/planes/wall",
                vertices=wall_corners,
                faces=faces.copy(),
                color=(0.0, 0.0, 1.0),  # red
                opacity=0.98,
                side="double",
                cast_shadow=False,
                receive_shadow=False
            )
            extra_handles.extend([fp_h, wp_h])
            for h in extra_handles:
                added_handles[h.name] = h

        return added_handles

    def to_viser_scene(self, scene: viser.SceneApi) -> Dict[str, viser.SceneNodeHandle]:
        bg_category, *bg_kind = self.bg_type.split(':')
        return getattr(self, f'_to_viser_scene_{bg_category}')(scene, *bg_kind)

    @classmethod
    def from_capturestudio_dataset(cls,
                                   dataset: MultiSessionDataset,
                                   t: int = 0,
                                   estimator_cls: Type[CapturestudioVirtualBackgroundFloorWallEstimator] = CapturestudioVirtualBackgroundFloorWallEstimatorViser,
                                   **estimator_kwargs) -> 'CapturestudioVirtualBackgroundViser':
        bg = super().from_capturestudio_dataset(dataset, t, estimator_cls, **estimator_kwargs)
        bg.__class__ = cls
        return bg

    @classmethod
    def from_object(cls, *args, **kwargs) -> 'CapturestudioVirtualBackgroundViser':
        bg = super().from_object(*args, **kwargs)
        bg.__class__ = cls
        return bg


# if __name__ == '__main__':
#     server_ = viser.ViserServer(host='0.0.0.0', port=8084)
#     server_.gui.configure_theme(control_layout="floating", dark_mode=False)
#     ply_file_ = PathUtils.resources_path() / 'backdrop_objects/rockstadt8k_PostShot/rockstadt_gs.ply'
#
#     bg = CapturestudioVirtualBackgroundViser.from_object(str(ply_file_), is_gs=True)
#     bg.to_viser_scene(server_.scene)
#
#     print("Viser server running at http://localhost:8084")
#     print("Press Ctrl+C to quit.")
#
#     # Wait loop to keep the main thread alive
#     try:
#         while True:
#             time.sleep(1.0)
#     except KeyboardInterrupt:
#         print("\nShutting down Viser server...")
#
#     exit(0)


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualCamerasViser(CapturestudioVirtualCameras):
    def tick(self, handles: Dict[str, viser.SceneNodeHandle]) -> Optional[Tuple[int, int]]:
        super_out = super().tick()
        if super_out is not None:
            t_prev, t_next = super_out

            def _subsample(_n: int, _cap: int = 32) -> list[int]:
                if _n <= _cap:
                    return list(range(_n))
                return list(np.round(np.linspace(0, _n - 1, _cap)).astype(int))

            def _nearest_in(_ids: list[int], _target: int) -> int:
                if not _ids:
                    return _target
                _arr = np.asarray(_ids, np.int32)
                return int(_arr[np.argmin(np.abs(_arr - _target))])

            # Style constants (match to_viser_scene)
            col_gt_inact, col_gt_act = (0.08, 0.08, 0.08), (0.0, 1.0, 0.0)
            lw_gt_inact, lw_gt_act = 1.6, 3.2  # doubled when active
            col_v_inact, col_v_act = (0.85, 0.85, 0.85), (1.0, 0.0, 0.0)
            lw_v_inact, lw_v_act = 1.3, 2.6

            nV = len(self.virtual_extrinsics_c2ws)
            nG = int(self.gt_extrinsics_c2ws.shape[0]) if self.gt_extrinsics_c2ws is not None else 0

            # ---- GT highlight update ----
            if self.camera_show_gt_frusta and nG > 0 and len(self.virtual_assignment) > 0:
                idxs_gt = _subsample(nG, 32)
                prev_gt = int(self.virtual_assignment[min(t_prev, len(self.virtual_assignment) - 1)])
                next_gt = int(self.virtual_assignment[min(t_next, len(self.virtual_assignment) - 1)])
                prev_gt_s = _nearest_in(idxs_gt, prev_gt)
                next_gt_s = _nearest_in(idxs_gt, next_gt)

                # Unhighlight previous
                h_prev = handles.get(f"gt:{prev_gt_s}")
                if h_prev is not None:
                    h_prev.color = col_gt_inact
                    h_prev.line_width = lw_gt_inact

                # Highlight next
                h_next = handles.get(f"gt:{next_gt_s}")
                if h_next is not None:
                    h_next.color = col_gt_act
                    h_next.line_width = lw_gt_act

            # ---- Virtual highlight update ----
            if self.camera_show_virtual_frusta and nV > 0:
                idxs_v = _subsample(nV, 32)
                prev_v_s = _nearest_in(idxs_v, t_prev)
                next_v_s = _nearest_in(idxs_v, t_next)

                h_prev_v = handles.get(f"virt:{prev_v_s}")
                if h_prev_v is not None:
                    h_prev_v.color = col_v_inact
                    h_prev_v.line_width = lw_v_inact

                h_next_v = handles.get(f"virt:{next_v_s}")
                if h_next_v is not None:
                    h_next_v.color = col_v_act
                    h_next_v.line_width = lw_v_act

        return super_out

    def to_viser_scene(self,
                       scene: viser.SceneApi,
                       R_scene: Optional[np.ndarray] = None,
                       t_scene: Optional[np.ndarray] = None,
                       initial_camera: Optional[viser.InitialCameraConfig] = None) -> Dict[str, viser.SceneNodeHandle]:
        # ---------- helpers ----------
        def _subsample(n: int, cap: int = 32) -> list[int]:
            if n <= cap: return list(range(n))
            return list(np.round(np.linspace(0, n - 1, cap)).astype(int))

        def _fovy_aspect_from_K(_K: np.ndarray) -> tuple[float, float]:
            _K = np.asarray(_K, np.float64)
            _fov_y = 2.0 * np.arctan2(self.image_size_hw[0], 2.0 * max(float(_K[1, 1]), 1e-6))
            _aspect = float(self.image_size_hw[1]) / float(self.image_size_hw[0])
            return float(_fov_y), float(_aspect)

        def _apply_scene(_c2w: np.ndarray) -> np.ndarray:
            if R_scene is None or t_scene is None:
                return _c2w
            _Rw = _c2w[:3, :3]
            _tw = _c2w[:3, 3]
            _out = np.eye(4, dtype=np.float64)
            _out[:3, :3] = R_scene @ _Rw
            _out[:3, 3] = (R_scene @ _tw) + t_scene
            return _out

        def _add_frustum(
                _name: str,
                _K: np.ndarray,
                _c2w: np.ndarray,
                *,
                _color=(0.08, 0.08, 0.08),
                _line_width: float = 1.5,
                _highlight: bool = False,
                _scale: float = 0.1,
                _variant: str = "wireframe",
        ):
            _c2w_v = _apply_scene(_c2w)
            _R = _c2w_v[:3, :3]
            _t = _c2w_v[:3, 3]
            _wxyz = rotation_matrix_to_quaternion(torch.from_numpy(_R)).numpy()
            _pos = float(_t[0]), float(_t[1]), float(_t[2])
            _fov_y, _aspect = _fovy_aspect_from_K(_K)
            _lw = 2.0 * _line_width if _highlight else _line_width
            return scene.add_camera_frustum(
                name=_name,
                fov=_fov_y,
                aspect=_aspect,
                scale=_scale,
                line_width=_lw,
                color=_color,
                image=None,
                wxyz=_wxyz,
                position=_pos,
                visible=True,
                cast_shadow=False,
                receive_shadow=False,
                variant=_variant,
            )

        # ---------- draw ----------
        handles: Dict[str, viser.SceneNodeHandle] = {}
        scene.set_up_direction((0.0, -1.0, 0.0))  # Viser convention

        # Active indices
        t_idx = min(max(int(self._t_current), 0), max(0, len(self.virtual_extrinsics_c2ws) - 1))
        active_gt = int(self.virtual_assignment[t_idx]) if len(self.virtual_assignment) > 0 else -1

        # GT frusta
        if self.camera_show_gt_frusta and self.gt_extrinsics_c2ws is not None and len(self.gt_extrinsics_c2ws) > 0:
            idxs_gt = _subsample(self.gt_extrinsics_c2ws.shape[0], 32)
            for i in idxs_gt:
                is_active = (i == active_gt)
                h = _add_frustum(
                    _name=f"/cams/gt/{i}",
                    _K=self.gt_intrinsics[i],
                    _c2w=self.gt_extrinsics_c2ws[i],
                    _color=(0.0, 1.0, 0.0) if is_active else (0.08, 0.08, 0.08),
                    _line_width=1.6,
                    _highlight=is_active,
                    _scale=self.frustum_scale,
                )
                handles[f"gt:{i}"] = h

        # Virtual frusta
        if self.camera_show_virtual_frusta and self.virtual_extrinsics_c2ws is not None and len(self.virtual_extrinsics_c2ws) > 0:
            idxs_v = _subsample(len(self.virtual_extrinsics_c2ws), 32)
            # make sure caches are ready for consistent poses
            self._compute_lookat_if_needed(R_scene, t_scene)
            self._roll_align_first_to_gt(R_scene, t_scene)

            for i in idxs_v:
                is_active = (i == t_idx)
                c2w_prepared = self._prepare_virtual_c2w(i, R_scene, t_scene)
                R = c2w_prepared[:3, :3]
                t = c2w_prepared[:3, 3]
                wxyz = rotation_matrix_to_quaternion(torch.from_numpy(R)).numpy()
                pos = (float(t[0]), float(t[1]), float(t[2]))
                fov_y, aspect = _fovy_aspect_from_K(self.virtual_intrinsics[i])
                lw = 2.0 * 1.3 if is_active else 1.3
                h = scene.add_camera_frustum(
                    name=f"/cams/virt/{i}",
                    fov=fov_y, aspect=aspect, scale=self.frustum_scale,
                    line_width=lw,
                    color=(1.0, 0.0, 0.0) if is_active else (0.85, 0.85, 0.85),
                    image=None, wxyz=wxyz, position=pos,
                    visible=True, cast_shadow=False, receive_shadow=False, variant="wireframe",
                )
                handles[f"virt:{i}"] = h

        # Snap viewer to first virtual camera
        if initial_camera is not None and len(self.virtual_extrinsics_c2ws) > 0:
            c2w0 = self._prepare_virtual_c2w(0, R_scene, t_scene)
            R0, t0 = c2w0[:3, :3], c2w0[:3, 3]
            initial_camera.wxyz = rotation_matrix_to_quaternion(torch.from_numpy(R0)).numpy()
            initial_camera.position = (float(t0[0]), float(t0[1]), float(t0[2]))
            fovy0, _ = _fovy_aspect_from_K(self.virtual_intrinsics[0])
            initial_camera.fov = float(fovy0)

        self._R_scene = R_scene
        self._t_scene = t_scene
        return handles

    @classmethod
    def from_capturestudio_dataset(cls, *args, **kwargs) -> 'CapturestudioVirtualCamerasViser':
        c = super().from_capturestudio_dataset(*args, **kwargs)
        c.__class__ = cls
        return c


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualDynamicForegroundViser(CapturestudioVirtualDynamicForeground):
    gs_cov_scale_viser: float = 1e4  # DO NOT CHANGE: it is hardcoded in viser's frontend code
    gs_scales_factor: float = 0.9  # manually tweak scales before cov3d computation (e.g. to increase sharpness)

    @functools.cached_property
    def label(self) -> str:
        return f'fg_{self.index:02d}'

    def tick(self, new_active_cam_idx: Optional[int] = None, target_w2c: Optional[np.nditer] = None, handles: Optional[Dict[str, 'viser.SceneNodeHandle']] = None) -> Dict[str, 'viser.SceneNodeHandle']:
        super().tick(new_active_cam_idx=new_active_cam_idx)
        self.to_viser_scene(*self._last_vis_args, target_w2c=target_w2c, handles=handles)

    def to_viser_scene(self, scene: viser.SceneApi, R_scene: Optional[np.ndarray] = None, t_scene: Optional[np.ndarray] = None, target_w2c: Optional[np.nditer] = None, handles: Optional[Dict[str, 'viser.SceneNodeHandle']] = None) -> Optional[Dict[str, 'viser.SceneNodeHandle']]:
        self._last_vis_args = (scene, R_scene, t_scene)
        image = next(self.image_generator)
        if isinstance(image, list):
            if self._blending_strategy == 'swap':
                image = image[self._active_cam_idx]
            elif self._blending_strategy == 'fuse':
                image = fuse_depth_maps(image)[self._active_cam_idx]
            elif self._blending_strategy.startswith('blend'):
                # BlendPCR-like blending
                if target_w2c is None:
                    # fallback to swap
                    log(f'[{self.__class__.__name__}::to_viser_scene] WARNING: target_w2c not provided, falling back to first camera', 'warning')
                    target_w2c = image[self._active_cam_idx].extrinsic_w2c
                closest_idx = [
                    max(0, self._active_cam_idx - 1),
                    self._active_cam_idx,
                    # min(len(image) - 1, self._active_cam_idx + 1),
                ]
                closest_idx = list(set(closest_idx))
                if len(closest_idx) == 3:
                    closest_idx = closest_idx[-2:]
                image = blend_point_cloud([image[i] for i in closest_idx], target_w2c=target_w2c, voxel_size_m=0.006, angle_power=4.0, refine_registration='deformation_pyramid')
            elif self._blending_strategy.startswith('merge') and self._blending_strategy != 'merge:naive':
                raise NotImplementedError

        label = f'{self.label}/{"gs" if isinstance(image, GSImage) else "pcd"}'
        if isinstance(image, GSImage):
            gs_3dgs = image.unproject().as_3dgs
            means3D = gs_3dgs['means3D'].detach().cpu().numpy().reshape(-1, 3)
            if R_scene is not None:
                means3D = np.ascontiguousarray((R_scene @ means3D.T).T)
            if t_scene is not None:
                means3D += t_scene

            from pytorch3d.transforms import quaternion_to_matrix
            R = quaternion_to_matrix(gs_3dgs['rotations'] / (gs_3dgs['rotations'].norm(dim=-1, keepdim=True) + 1e-12))
            if R_scene is not None:
                R = torch.from_numpy(R_scene).to(device=R.device, dtype=R.dtype).reshape(-1, 3, 3).transpose(-1, -2) @ R
            C = R @ torch.diag_embed((gs_3dgs['scales'] * self.gs_scales_factor) ** 2) @ R.transpose(-1, -2)
            C = C.detach().cpu().reshape(-1, 3, 3) * self.gs_cov_scale_viser
            rgbs = gs_3dgs['colors_precomp'].detach().cpu().numpy().reshape(-1, 3)
            opacities = gs_3dgs['opacities'].detach().cpu().numpy().reshape(-1, 1)
            rgbs = np.clip(rgbs, 0.0, 1.0).astype(np.float32)
            opacities = np.clip(opacities, 1e-5, 1.0).astype(np.float32)

            finite = (torch.isfinite(C).all(dim=(-1, -2)) & torch.isfinite(torch.from_numpy(means3D)).all(-1)
                      & torch.isfinite(torch.from_numpy(rgbs)).all(-1) & torch.isfinite(torch.from_numpy(opacities)).all(-1))
            finite &= (C.abs() > 0.000061).all(dim=(-1, -2)) & (C.abs() <= 2.0).all(dim=(-1, -2))
            finite = finite.numpy()
            # finite &= opacities.squeeze() > 0.00007

            # means3D = torch.rand_like(torch.from_numpy(means3D[finite])).numpy()
            means3D = means3D[finite]
            rgbs = rgbs[finite]
            opacities = opacities[finite]
            C = C.numpy()[finite]

            if handles is None or label not in handles:
                h = scene.add_gaussian_splats(
                    name=label,  # + str(Str.random(10)),
                    centers=means3D,
                    covariances=C,
                    rgbs=rgbs,
                    opacities=opacities,
                    visible=True
                )
            else:
                h = handles[label]
                h.visible = False
                h.centers = means3D
                h.covariances = C
                h.rgbs = rgbs
                h.opacities = opacities
                h.visible = True
            return {label: h}

        if isinstance(image, RGBDImage):
            image = image.unproject().open3d

        if isinstance(image, o3d.geometry.PointCloud):
            pcd = image
            pts = np.asarray(pcd.points, dtype=np.float64)
            cols = np.asarray(pcd.colors, dtype=np.float64)
            if R_scene is not None:
                pts = (R_scene @ pts.T).T
            if t_scene is not None:
                pts += t_scene
            if handles is None or label not in handles:
                h = scene.add_point_cloud(
                    name=label,
                    points=pts.astype(np.float32),
                    colors=np.clip(cols, 0.0, 1.0).astype(np.float32),
                    point_size=0.0035,
                    point_shape='square',
                )
            else:
                h = handles[label]
                h.visible = False
                h.points = pts.astype(np.float32)
                h.colors = np.clip(cols, 0.0, 1.0).astype(np.float32)
                h.visible = True
            return {label: h}

        raise NotImplementedError('Unexpected type:', type(image))

    @classmethod
    def from_capturestudio_dataset(cls, *args, **kwargs) -> 'CapturestudioVirtualDynamicForegroundViser':
        fg = super().from_capturestudio_dataset(*args, **kwargs)
        fg.__class__ = cls
        return fg

    @classmethod
    def from_merged_ply_files(cls, *args, **kwargs) -> 'CapturestudioVirtualDynamicForegroundViser':
        fg = super().from_merged_ply_files(*args, **kwargs)
        fg.__class__ = cls
        return fg


class CaptureStudioVirtualSceneViser(CapturestudioVirtualScene):
    def __init__(self, viser_scene: viser.SceneApi, initial_camera: viser.InitialCameraConfig, *args, **kwargs):
        self._viser_scene = viser_scene
        self.initial_camera = initial_camera
        self._all_handles = {}
        super().__init__(*args, **kwargs)

    @functools.cached_property
    def server(self) -> viser.ViserServer:
        owner = getattr(self._viser_scene, "_owner", None)  # ViserServer or ClientHandle
        if owner is None or not hasattr(owner, "get_clients"):
            raise RuntimeError("Cannot locate ViserServer from SceneApi (no _owner.get_clients).")
        return owner

    @property
    def client(self) -> Optional[viser.ClientHandle]:
        owner = getattr(self._viser_scene, "_owner", None)  # ViserServer or ClientHandle
        if owner is None or not hasattr(owner, "get_clients"):
            raise RuntimeError("Cannot locate ViserServer from SceneApi (no _owner.get_clients).")
        clients_dict = owner.get_clients()
        if len(clients_dict) == 0:
            return None
        client: viser.ClientHandle = next(iter(clients_dict.values()))
        return client

    def capture(self, target_size_hw: Tuple[int, int], show_background: bool = True) -> np.ndarray:
        reshow_handles = []
        if not show_background:
            for hk, hv in self._all_handles.items():
                if hk.startswith('background') and hv.visible:
                    hv.visible = False
                    time.sleep(0.2)
                    reshow_handles.append(hv)

        cam = self._cameras.current_camera
        K = np.asarray(cam["intrinsics"], np.float64)
        c2w = np.asarray(cam["extrinsic_c2w"], np.float64)

        # Pose + intrinsics → render params
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        wxyz = rotation_matrix_to_quaternion(torch.from_numpy(R)).numpy()
        H, W = target_size_hw
        fy = float(K[1, 1])
        fov_y = 2.0 * float(np.arctan(H / (2.0 * fy)))
        # print('H', H, 'W', W, 'fy', fy, 'fov_y', fov_y)

        # Get a connected client from the server owning this SceneApi.
        client = self.client
        client.flush()

        # Ask the client to render off-screen under our virtual camera
        rgb = client.get_render(
            height=int(H),
            width=int(W),
            wxyz=wxyz,
            position=(float(t[0]), float(t[1]), float(t[2])),
            fov=fov_y,
            transport_format="jpeg",
        )

        if not show_background:
            for hv in reshow_handles:
                hv.visible = True
                time.sleep(0.2)

        return rgb

    def on_scene_updated(self, what_updated: str, set_not_unset: bool = True) -> None:
        """
        Updated viewer once scene elements have been updated.

        Parameters
        ----------
        what_updated: str
            One or more of 'foregrounds', 'background', 'cameras', 'lights'. If more than one, they will be joined with "|".
        set_not_unset: bool
            If True, the element(s) were set, otherwise they were unset.
        """
        if set_not_unset:
            kwargs = {}
            if what_updated != 'background':
                kwargs = dict(R_scene=self._R_scene, t_scene=self._t_scene)
            if what_updated == 'foregrounds' and self._cameras is not None:
                kwargs['target_w2c'] = np.linalg.inv(self._cameras.current_camera['extrinsic_c2w'])
            if what_updated == 'cameras':
                kwargs['initial_camera'] = self.initial_camera
            attr = getattr(self, f'_{what_updated}')
            if not isinstance(attr, list):
                attr = [attr]
            for i, attr_i in enumerate(attr):
                added_handles = attr_i.to_viser_scene(self._viser_scene, **kwargs)
                for hk, hv in added_handles.items():
                    if not hk.startswith(f'{what_updated}:'):
                        hk = f'{what_updated}:{i:02d}:{hk}'
                    self._all_handles[hk] = hv
        else:
            for hk in list(self._all_handles.keys()):
                if hk.startswith(f'{what_updated}:'):
                    self._all_handles[hk].remove()
                    time.sleep(0.2)
                    del self._all_handles[hk]

    def flush(self) -> 'CaptureStudioVirtualSceneViser':
        if self.server:
            self.server.flush()
            time.sleep(0.5)
        if self.client:
            self.client.flush()
            time.sleep(0.5)
        return self

    def tick(self):
        # updated scene elements
        for _asset in ['background', 'cameras', 'foregrounds', 'lights']:
            attr = getattr(self, f'_{_asset}')
            if attr is None or not hasattr((attr if not isinstance(attr, list) else attr[0]), 'tick'):
                continue
            if not isinstance(attr, list):
                attr = [attr]
            for i, attr_i in enumerate(attr):
                attr_i_handles = {
                    hk.split(':')[-1]: hv
                    for hk, hv in self._all_handles.items()
                    if hk.startswith(f'{_asset}:{i:02d}:')
                }
                kwargs = dict(handles=attr_i_handles)
                if _asset == 'foregrounds' and self._cameras is not None:
                    kwargs['new_active_cam_idx'] = self._cameras.gt_cam_index
                    kwargs['target_w2c'] = np.linalg.inv(self._cameras.current_camera['extrinsic_c2w'])
                attr_i.tick(**kwargs)

        # sync updates
        self.flush()

    def reset(self):
        self._viser_scene.reset()
        for _asset in ['background', 'cameras', 'foregrounds', 'lights']:
            attr = getattr(self, f'_{_asset}')
            if attr is None:
                continue

            getattr(self, f'set_{_asset}')(attr)

    @classmethod
    def from_capturestudio_session(cls,
                                   dataset_raw: MultiSessionDataset,
                                   dataset_vis: Optional[Union[MultiSessionDataset, List[MultiSessionDataset], Path, List[Path]]] = None,
                                   wall_overshoot_m: float = 1.3,
                                   use_gs: bool = False,
                                   t_start: Union[int, List[int]] = 0,
                                   t_total: Union[int, List[int]] = -1,
                                   camera_orbit_type: Literal['interpolated', 'audience'] = 'audience',
                                   write_ply_files: bool = False,
                                   **kwargs) -> 'CaptureStudioVirtualSceneViser':
        if dataset_vis is None:
            dataset_vis = dataset_raw
        if not isinstance(dataset_vis, list):
            dataset_vis = [dataset_vis]
        if not isinstance(t_start, list):
            t_start = [t_start] * len(dataset_vis)
        if not isinstance(t_total, list):
            t_total = [t_total] * len(t_start)
        floor_wall_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith('floor') or k.startswith('wall')}
        bg_kwargs = {k[3:]: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith('bg_')}
        fg_kwargs = {k[3:]: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith('fg_')}
        if 'obj_path' in bg_kwargs:
            # ply bg
            scene_background = CapturestudioVirtualBackgroundViser.from_object(**bg_kwargs)
        else:
            # floor + wall bg
            scene_background = CapturestudioVirtualBackgroundViser.from_capturestudio_dataset(dataset_raw, wall_overshoot_m=wall_overshoot_m, **floor_wall_kwargs)
        scene_cameras = CapturestudioVirtualCamerasViser.from_capturestudio_dataset(
            dataset_raw,
            background=scene_background,
            camera_orbit_type=camera_orbit_type,
            t_total=max(t_total),
            **{k: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith('camera')}
        )
        if isinstance(dataset_vis[0], MultiSessionDataset):
            assert len(dataset_vis) == len(t_start) == len(t_total), f'Argument length mismatch: {len(dataset_vis)} vs {len(t_start)} vs {len(t_total)}'
            scene_foregrounds = [
                CapturestudioVirtualDynamicForegroundViser.from_capturestudio_dataset(
                    dataset_vis_i,
                    use_gs=use_gs,
                    t_start=t_start_i,
                    t_total=t_total_i,
                    index=i,
                    write_ply_files=write_ply_files,
                    active_cam_idx=scene_cameras.gt_cam_index,
                    **fg_kwargs
                )
                for i, (dataset_vis_i, t_start_i, t_total_i) in enumerate(zip(dataset_vis, t_start, t_total))
            ]
        elif isinstance(dataset_vis[0], Path):
            scene_foregrounds = [
                CapturestudioVirtualDynamicForegroundViser.from_merged_ply_files(
                    ply_root=dataset_vis_i,
                    ply_format='{ply_index:06d}.ply',
                    t_start=t_start_i,
                    t_total=t_total_i,
                    index=i,
                )
                for i, (dataset_vis_i, t_start_i, t_total_i) in enumerate(zip(dataset_vis, t_start, t_total))
            ]
        else:
            raise NotImplementedError
        return (
            cls(**kwargs)
            .set_background(background=scene_background)
            .set_foregrounds(foregrounds=scene_foregrounds)
            .set_cameras(cameras=scene_cameras)
        ).flush()


class TeaserGeneratorViser(TeaserGenerator):
    def __init__(
            self,
            headless: bool = True,
            headless_use_egl: bool = True,
            headless_dpr: float = 1.0,
            server_host: str = '0.0.0.0',
            server_port: int = 8088,
            show_background: bool = True,
            *args,
            **kwargs,
    ):
        self.headless = headless
        self.headless_use_egl = headless_use_egl
        self.headless_dpr = headless_dpr
        self.server_host = server_host
        self.server_port = server_port
        self._show_background = show_background
        self._server = None
        self._viser_start()
        super().__init__(*args, **kwargs)

    def _init_scene(self):
        if self.t_total == -1:
            self.t_total = min(len(_) for _ in self.datasets_vis)
        self._scene = CaptureStudioVirtualSceneViser.from_capturestudio_session(
            dataset_raw=self.dataset_raw,
            dataset_vis=self.datasets_vis,
            viser_scene=self._server.scene,
            initial_camera=self._server.initial_camera,
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

    def _viser_start(self):
        self._server = viser.ViserServer(host='0.0.0.0', port=8084)
        self._server.gui.configure_theme(control_layout="floating", dark_mode=False)
        self.server_host = str(self._server.get_host())
        self.server_port = str(self._server.get_port())

    def _viser_stop(self):
        self._server.scene.reset()
        self._server.stop()

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
            video_writer_.write(cv2.cvtColor(rendered_img_, cv2.COLOR_RGB2BGR))

            self._scene.tick()
            time.sleep(1.0)

        if video_writer_ is not None:
            video_writer_.release()
            log(f'[{self.__class__.__name__}::_rendering_loop] Video file released: "{self.out_video_path}"')

        return self.out_video_path

    @property
    def _run_headless(self) -> Path:
        # -------------------------------------------
        # HEADLESS MODE (Playwright only to create a client;
        # frames are captured via virtual_scene_.capture())
        # -------------------------------------------
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            args = [
                "--enable-webgl", "--ignore-gpu-blocklist",
                "--no-sandbox", "--disable-dev-shm-usage",
            ]
            if self.headless_use_egl:
                # GPU ON + EGL for headless WebGL
                args += ["--use-gl=egl"]
            else:
                # Software fallback only if you must
                args += ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"]
            browser = p.chromium.launch(
                headless=True,
                args=args,
            )

            context = browser.new_context(
                screen={"width": self.render_config.image_size_hw[1], "height": self.render_config.image_size_hw[0]},
                viewport={"width": self.render_config.image_size_hw[1], "height": self.render_config.image_size_hw[0]},
                device_scale_factor=self.headless_dpr,
                java_script_enabled=True,
                ignore_https_errors=True,
                is_mobile=False
            )
            context.add_init_script(f"""
            (() => {{
              const style = document.createElement('style');
              style.type = 'text/css';
              style.innerHTML = `
                html, body, #root {{ margin:0; padding:0; width:100vw; height:100vh; overflow:hidden; }}
                .viser-gui, #gui-root, .stats-container {{ display:none !important; }}
              `;
              document.documentElement.appendChild(style);
              // Ensure DPR is what you expect
              Object.defineProperty(window, 'devicePixelRatio', {{ get: () => {self.headless_dpr} }});
              // Log crash errors
              window.addEventListener('webglcontextlost', e => console.warn('WebGL context lost', e));
              window.addEventListener('error', e => console.error('window error', e.message));
            }})();
            """)

            page = context.new_page()
            page.goto(f'http://127.0.0.1:{self._server.get_port()}?fixedDpr={self.headless_dpr}', wait_until='domcontentloaded')
            # Wait for the GL canvas to exist and match the requested size.
            page.wait_for_selector("canvas[data-engine]", state="visible")
            page.wait_for_function(
                "(arg) => { "
                "  const [H, W] = arg; "
                "  const c = document.querySelector('canvas[data-engine]'); "
                "  return c && c.clientWidth === W && c.clientHeight === H && "
                "         c.width === W && c.height === H; "
                "}",
                arg=self.render_config.image_size_hw,
                timeout=15000
            )
            gl_canvas = page.locator("canvas[data-engine]").last
            info = gl_canvas.evaluate("""
                                      (c) => {
                                          const gl = c.getContext('webgl2') || c.getContext('webgl');
                                          return {
                                              dpr: window.devicePixelRatio,
                                              client: {w: c.clientWidth, h: c.clientHeight},
                                              attr: {w: c.width, h: c.height},
                                              db: gl ? {w: gl.drawingBufferWidth, h: gl.drawingBufferHeight} : null
                                          };
                                      }
                                      """)
            log(f'[__main__] Client setup finished. Metrics: {info}', 'info')

            # Warmup RAFs
            page.evaluate("() => new Promise(r => requestAnimationFrame(()=>requestAnimationFrame(r)))")

            # Wait until a client is registered on the server (our headless Chromium)
            t0 = time.time()
            while time.time() - t0 < 10.0:
                if len(self._server.get_clients()) > 0:
                    break
                time.sleep(0.05)
            if len(self._server.get_clients()) == 0:
                raise RuntimeError("No Viser client connected in headless mode.")

            # Render loop: use the same Python-side capture() as interactive mode
            p = self._rendering_loop()

            context.close()
            browser.close()

        # shutdown
        self._viser_stop()

        return p

    def _run_interactive(self) -> Path:
        # -------------------------------------------
        # INTERACTIVE MODE
        # -------------------------------------------
        p = self._rendering_loop()

        try:
            while True:
                time.sleep(1.0)
        finally:
            self._viser_stop()
            return p

    def run(self) -> Path:
        if self.headless:
            p = self._run_headless
        else:
            p = self._run_interactive()
        return p

    @classmethod
    def generate_human_eval_videos(cls, pcd_path: str, gs_path: str, out_video_path: str = '{pcd_or_gs_first}_{pcd_or_gs_second}.mp4', fps: int = 30) -> Tuple[str, str]:
        # --------------------------------------
        #  Create human eval videos
        # -------------------------------------
        from moviepy.editor import VideoFileClip, clips_array
        from moviepy.video.fx.crop import crop
        from moviepy.video.fx.resize import resize

        CROP_TOP_PX = 28
        CROP_BOTTOM_PX = 100
        CROP_LEFT_PX = 0
        CROP_RIGHT_PX = 0

        def _crop_unpad_px(
                clip: VideoFileClip,
                top_px: int = 0,
                bottom_px: int = 0,
                left_px: int = 0,
                right_px: int = 0,
        ) -> VideoFileClip:
            """Crop fixed pixels from each side. Guards against invalid crops."""
            w, h = clip.size

            # Clamp to avoid over-cropping
            top_px = max(0, min(int(top_px), h - 2))
            bottom_px = max(0, min(int(bottom_px), h - 2 - top_px))
            left_px = max(0, min(int(left_px), w - 2))
            right_px = max(0, min(int(right_px), w - 2 - left_px))

            y1 = top_px
            y2 = h - bottom_px
            x1 = left_px
            x2 = w - right_px

            # Final guard (must keep at least 2x2 px)
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                # Fall back to no-op crop rather than crashing
                return clip

            return crop(clip, x1=x1, y1=y1, x2=x2, y2=y2)

        def _make_side_by_side(left_path: str, right_path: str, out_path: str):
            left = VideoFileClip(left_path, audio=False)
            right = VideoFileClip(right_path, audio=False)
            final = None

            try:
                # Ensure same duration (avoid desync / last-frame freeze)
                min_dur = min(left.duration, right.duration)
                left = left.subclip(0, min_dur)
                right = right.subclip(0, min_dur)

                # Fixed crop (top/bottom) on BOTH videos
                left = _crop_unpad_px(
                    left,
                    top_px=CROP_TOP_PX,
                    bottom_px=CROP_BOTTOM_PX,
                    left_px=CROP_LEFT_PX,
                    right_px=CROP_RIGHT_PX,
                )
                right = _crop_unpad_px(
                    right,
                    top_px=CROP_TOP_PX,
                    bottom_px=CROP_BOTTOM_PX,
                    left_px=CROP_LEFT_PX,
                    right_px=CROP_RIGHT_PX,
                )

                # Match heights for clean horizontal concatenation
                target_h = min(left.h, right.h)
                if left.h != target_h:
                    left = resize(left, height=target_h)
                if right.h != target_h:
                    right = resize(right, height=target_h)

                final = clips_array([[left, right]])

                # Write (yuv420p for broad compatibility)
                final.write_videofile(
                    out_path,
                    codec="libx264",
                    audio=False,
                    fps=fps,
                    preset="medium",
                    ffmpeg_params=["-pix_fmt", "yuv420p"],
                )
            finally:
                # Always close resources
                left.close()
                right.close()
                if final is not None:
                    final.close()

        OUT_PCD_LEFT = out_video_path.format(pcd_or_gs_first='PCD', pcd_or_gs_second='GS')
        OUT_GS_LEFT = out_video_path.format(pcd_or_gs_first='GS', pcd_or_gs_second='PCD')

        _make_side_by_side(pcd_path, gs_path, OUT_PCD_LEFT)  # PCD | GS
        _make_side_by_side(gs_path, pcd_path, OUT_GS_LEFT)  # GS  | PCD

        log(f"[{cls.__name__}::create_human_eval_videos] Human-eval videos written:\n  - {OUT_PCD_LEFT}\n  - {OUT_GS_LEFT}", 'info')
        return OUT_PCD_LEFT, OUT_GS_LEFT

    @staticmethod
    def stitch_performances(*perf_dirs: Path, output_dir: Path):
        # Assuming cameras cam01 through cam07 based on your output
        cameras = [d.name for d in sorted(list(perf_dirs[0].glob('cam*')), key=lambda x: int(x.name.replace('cam', ''))) if d.is_dir()]
        for cam in cameras:
            cam_dirs = [pd / cam for pd in perf_dirs]
            assert all(cd.exists() for cd in cam_dirs)

            out_cam_dir = output_dir / cam
            out_cam_dir.mkdir(parents=True, exist_ok=True)

            # Iterate through all ply files in the first performance
            for ply_file1 in tqdm(sorted(cam_dirs[0].glob("*.ply"), key=lambda x: int(x.stem)), total=len(list(cam_dirs[0].glob("*.ply")))):
                filename = ply_file1.name
                ply_files = [cd / ply_file1.name for cd in cam_dirs]
                assert all(f.exists() for f in ply_files)

                # Load the point clouds
                # Note: Open3D expects string paths, so we cast the Path objects
                pcds = [o3d.io.read_point_cloud(str(f)) for f in ply_files]

                # Merge the point clouds
                merged_pcd = sum(pcds, start=o3d.geometry.PointCloud())

                # Save the result
                out_path = out_cam_dir / filename
                o3d.io.write_point_cloud(str(out_path), merged_pcd)

        print("\nStitching complete!")


# if __name__ == '__main__':
#     TeaserGeneratorViser.stitch_performances(
#         Path("/mnt/d/TEASER_PCD/cagliari_1_perf_5"),
#         Path("/mnt/d/TEASER_PCD/cagliari_1_perf_7"),
#         output_dir=Path("/mnt/d/TEASER_PCD/cagliari_1_perf_5_plus_7")
#     )
#     exit(0)

if __name__ == '__main__':
    # @formatter:off
    DATA = [
        # ('Thanos_2_Perf_2',     'Thanos_2_Calib_1',     169,  [4, 5, 7, 8, 9]),
        # ('Simone_2_Perf_3',     'Thanos_2_Calib_1',     0,   [4, 5, 7, 8, 9]),
        # ('Nathalie_1_Perf_3',   'Thanos_2_Calib_1',     69,   [4, 5, 7, 8, 9]),
        # ('Den_1_Perf_2',        'Thanos_2_Calib_1',     69,   [4, 5, 7, 8, 9]),
        # ('Philipp_1_Perf_5',    'Philipp_1_Calib_1',    69,   [4, 5, 7, 8, 9]),
        # ('Philipp_1_Perf_6',    'Philipp_1_Calib_1',    69,   [4, 5, 7, 8, 9]),
        # ('Cagliari_1_Perf_5',    'Cagliari_1_Calib_6',    0,    list(range(1, 8))),
        # ('Cagliari_1_Perf_5|Cagliari_1_Perf_7',    'Cagliari_1_Calib_6',    0,    list(range(1, 8))),
        ('Cagliari_1_Perf_7',    'Cagliari_1_Calib_6',    0,    list(range(1, 8))),
        # ('Cagliari_1_Perf_7',    'Cagliari_1_Calib_6',    0,    list(range(1, 8))),
        # (['Simone_2_Perf_3', Path('/mnt/d/CAPTURESTUDIO_CACHE_BACKUP/Captures_Apr_May_2025/Simone_2_Perf_3/merging/concatenate_000000_000300')],    'Thanos_2_Calib_1',    0,   [4, 5, 7, 8, 9]),
        # (['Simone_2_Perf_3', Path('/mnt/d/CAPTURESTUDIO_CACHE_BACKUP/Captures_Apr_May_2025/Simone_2_Perf_3/merging/concatenate_outliers_000000_000300')],    'Thanos_2_Calib_1',    0,   [4, 5, 7, 8, 9]),
        # (['Simone_2_Perf_3', Path('/mnt/d/CAPTURESTUDIO_CACHE_BACKUP/Captures_Apr_May_2025/Simone_2_Perf_3/merging/tsdf_000000_000300')],    'Thanos_2_Calib_1',    0,   [4, 5, 7, 8, 9]),
        # (['Simone_2_Perf_3', Path('/mnt/d/CAPTURESTUDIO_CACHE_BACKUP/Captures_Apr_May_2025/Simone_2_Perf_3/merging/tsdf_000000_000300')],    'Thanos_2_Calib_1',    0,   [4, 5, 7, 8, 9]),
        # (['Simone_2_Perf_3', Path('/mnt/d/CAPTURESTUDIO_CACHE_BACKUP/Captures_Apr_May_2025/Simone_2_Perf_3/merging/voxels_outliers_000000_000300')],    'Thanos_2_Calib_1',    0,   [4, 5, 7, 8, 9]),
        # (['Simone_2_Perf_3', Path('/mnt/d/CAPTURESTUDIO_CACHE_BACKUP/Captures_Apr_May_2025/Simone_2_Perf_3/merging/LNDP_outliers_000000_000300')],    'Thanos_2_Calib_1',    0,   [4, 5, 7, 8, 9]),
    ]
    # @formatter:on
    DEBUG = False

    import time as time_
    import gc as gc_

    for SESSION_PERF, SESSION_CALIB, T_START, CAM_IDX in DATA:
        video_paths_ = {}
        for use_gs_ in [False]:
            visualizer_ = TeaserGeneratorViser(
                session_perf=SESSION_PERF if isinstance(SESSION_PERF, list) else SESSION_PERF.split('|'),
                session_calib=SESSION_CALIB,
                calib_method='MultiCamCalib',
                depth_source='bilateral_temporal',
                cam_idx_perf=CAM_IDX,
                cam_idx_raw=CAM_IDX,
                render_config=TeaserGeneratorRenderConfig.for_apr_may_2025(
                    use_gs=use_gs_,
                    image_size_hw=(1080, 1920),
                    # image_size_hw=(2560, 2048)
                    camera_traverse_velocity=0.2,
                    camera_orbit_offset_m=0.6,
                ),
                t_start=T_START,
                t_total=1 if DEBUG else -1,
                headless=not DEBUG,
                write_ply_files=False,
                debug=DEBUG,
                show_background=True,
                # camera_orbit_start_idx=0 if not DEBUG else 69,
                camera_orbit_start_idx=T_START,
                # # NEW: static bg
                # bg_obj_path=PathUtils.resources_path() / 'backdrop_objects/rockstadt8k_PostShot/rockstadt_gs.ply',
                # bg_is_gs=True,
                # NEW: blending of FG PCDs using BlendPCR style
                fg_blending_strategy='blend:blend_pcr'
                # fg_blending_strategy='fuse'
                # fg_blending_strategy='swap'
            )
            video_path_ = visualizer_.run()
            video_paths_[f"{'gs' if use_gs_ else 'pcd'}_path"] = video_path_

        # # Generate final videos
        # TeaserGeneratorViser.generate_human_eval_videos(
        #     **video_paths_,
        #     fps=30,
        #     out_video_path='+'.join([_.lower() for _ in visualizer_.sessions_perf]) + '_{pcd_or_gs_first}_{pcd_or_gs_second}.mp4'
        # )

        # Cleanup
        del visualizer_
        gc_.collect()
        torch.cuda.empty_cache()
        time_.sleep(5.0)
