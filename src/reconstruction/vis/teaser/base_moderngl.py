import dataclasses
import datetime
import functools
from pathlib import Path
from typing import Tuple, Optional, Literal, List, Union, Dict

import cv2
import moderngl
import numpy as np
import open3d as o3d
import torch
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
    TeaserGeneratorRenderConfig,
)
from utils.misc import log


# ---------------------------------------------------------------------------
# ModernGL renderable state
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class MGLRenderable:
    vao: moderngl.VertexArray
    vbo: moderngl.Buffer
    prog: moderngl.Program
    texture: Optional[moderngl.Texture] = None
    ibo: Optional[moderngl.Buffer] = None
    mode: int = moderngl.POINTS
    point_size: float = 3.0
    line_width: float = 1.0
    visible: bool = True
    num_vertices: int = 0

    def release(self) -> None:
        for obj in (self.vao, self.vbo, self.ibo, self.texture):
            if obj is None:
                continue
            try:
                obj.release()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Shader definitions
# ---------------------------------------------------------------------------
_PCD_VERTEX_SHADER = """
#version 330 core
in vec3 in_pos;
in vec3 in_color;
uniform mat4 u_mvp;
uniform float u_point_size;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    gl_PointSize = u_point_size;
    v_color = in_color;
}
"""

_PCD_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
out vec4 f_color;
void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    if (dot(p, p) > 1.0) discard;
    f_color = vec4(v_color, 1.0);
}
"""

_TEX_QUAD_VERTEX_SHADER = """
#version 330 core
in vec3 in_pos;
in vec2 in_uv;
uniform mat4 u_mvp;
out vec2 v_uv;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_uv = in_uv;
}
"""

_TEX_QUAD_FRAGMENT_SHADER = """
#version 330 core
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_tex, v_uv);
}
"""

_LINE_VERTEX_SHADER = """
#version 330 core
in vec3 in_pos;
in vec3 in_color;
uniform mat4 u_mvp;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_color = in_color;
}
"""

_LINE_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""

_MESH_VERTEX_SHADER = """
#version 330 core
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_normal;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_normal = normalize((u_model * vec4(in_normal, 0.0)).xyz);
    v_color = in_color;
}
"""

_MESH_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec3 v_color;
out vec4 f_color;
void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(vec3(0.35, -0.2, 0.9));
    float ndl = max(dot(N, L), 0.0);
    vec3 color = v_color * (0.2 + 0.8 * ndl);
    f_color = vec4(color, 1.0);
}
"""

_MESH_TEX_VERTEX_SHADER = """
#version 330 core
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_normal;
out vec2 v_uv;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_normal = normalize((u_model * vec4(in_normal, 0.0)).xyz);
    v_uv = in_uv;
}
"""

_MESH_TEX_FRAGMENT_SHADER = """
#version 330 core
uniform sampler2D u_tex;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 tex = texture(u_tex, v_uv);
    vec3 N = normalize(v_normal);
    vec3 L = normalize(vec3(0.35, -0.2, 0.9));
    float ndl = max(dot(N, L), 0.0);
    vec3 color = tex.rgb * (0.2 + 0.8 * ndl);
    f_color = vec4(color, tex.a);
}
"""


# ---------------------------------------------------------------------------
# Camera math
# Gemini's interpretation/path is intentionally preserved here.
# ---------------------------------------------------------------------------
def _get_projection_matrix(K: np.ndarray, W: int, H: int, near: float, far: float) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * fx / float(W)
    P[1, 1] = 2.0 * fy / float(H)
    P[0, 2] = (float(W) - 2.0 * cx) / float(W)
    P[1, 2] = -(float(H) - 2.0 * cy) / float(H)
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def _get_view_matrix(c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(c2w)
    cv2gl = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    return cv2gl @ w2c


def _load_image_rgba(path: Union[str, Path]) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGBA)
    elif arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    return np.flipud(arr)


def _apply_scene_tf(points: np.ndarray, R_scene: Optional[np.ndarray], t_scene: Optional[np.ndarray]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if R_scene is not None:
        pts = (np.asarray(R_scene, dtype=np.float32) @ pts.T).T
    if t_scene is not None:
        pts = pts + np.asarray(t_scene, dtype=np.float32).reshape(1, 3)
    return pts


# ---------------------------------------------------------------------------
# Pipeline classes
# ---------------------------------------------------------------------------
class CapturestudioVirtualBackgroundFloorWallEstimatorModernGL(CapturestudioVirtualBackgroundFloorWallEstimator):
    pass


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualBackgroundModernGL(CapturestudioVirtualBackground):
    def _make_quad_renderable(
        self,
        ctx: moderngl.Context,
        corners: np.ndarray,
        tex_path: Union[Path, str],
        R_scene: Optional[np.ndarray],
        t_scene: Optional[np.ndarray],
    ) -> Optional[MGLRenderable]:
        if corners is None or len(corners) < 4:
            return None
        c = _apply_scene_tf(np.asarray(corners[:4], dtype=np.float32), R_scene, t_scene)
        verts = np.array([c[0], c[1], c[2], c[0], c[2], c[3]], dtype=np.float32)
        uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.float32)
        vbo_data = np.hstack([verts, uvs]).astype(np.float32)

        prog = ctx.program(vertex_shader=_TEX_QUAD_VERTEX_SHADER, fragment_shader=_TEX_QUAD_FRAGMENT_SHADER)
        vbo = ctx.buffer(vbo_data.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "3f 2f", "in_pos", "in_uv")])

        img = _load_image_rgba(tex_path)
        tex = ctx.texture((img.shape[1], img.shape[0]), 4, img.tobytes())
        tex.build_mipmaps()
        tex.repeat_x = False
        tex.repeat_y = False
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        return MGLRenderable(
            vao=vao,
            vbo=vbo,
            prog=prog,
            texture=tex,
            mode=moderngl.TRIANGLES,
            num_vertices=6,
        )

    def _make_pointcloud_renderable(
        self,
        ctx: moderngl.Context,
        points: np.ndarray,
        colors: np.ndarray,
        point_size: float = 3.0,
    ) -> MGLRenderable:
        pts = np.asarray(points, dtype=np.float32)
        cols = np.asarray(colors, dtype=np.float32)
        if cols.max() > 1.0:
            cols = cols / 255.0
        vbo_data = np.hstack([pts, cols]).astype(np.float32)
        prog = ctx.program(vertex_shader=_PCD_VERTEX_SHADER, fragment_shader=_PCD_FRAGMENT_SHADER)
        vbo = ctx.buffer(vbo_data.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_pos", "in_color")])
        return MGLRenderable(
            vao=vao,
            vbo=vbo,
            prog=prog,
            mode=moderngl.POINTS,
            point_size=point_size,
            num_vertices=len(pts),
        )

    def _make_trimesh_renderable(self, ctx: moderngl.Context, mesh: trimesh.Trimesh) -> MGLRenderable:
        mesh = mesh.copy()
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if normals.shape != verts.shape:
            normals = np.zeros_like(verts, dtype=np.float32)
            normals[:, 2] = 1.0

        visual_kind = getattr(mesh.visual, "kind", None)
        if visual_kind == "texture" and getattr(mesh.visual, "uv", None) is not None:
            material = getattr(mesh.visual, "material", None)
            image = getattr(material, "image", None)
            if image is not None:
                image = np.asarray(image)
                if image.ndim == 2:
                    image = np.repeat(image[..., None], 3, axis=-1)
                if image.shape[2] == 3:
                    alpha = np.full((*image.shape[:2], 1), 255, dtype=image.dtype)
                    image = np.concatenate([image, alpha], axis=-1)
                image = np.flipud(image)
                uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
                interleaved = np.hstack([verts, normals, uvs]).astype(np.float32)
                vbo = ctx.buffer(interleaved.tobytes())
                ibo = ctx.buffer(faces.astype(np.int32).tobytes())
                prog = ctx.program(vertex_shader=_MESH_TEX_VERTEX_SHADER, fragment_shader=_MESH_TEX_FRAGMENT_SHADER)
                vao = ctx.vertex_array(
                    prog,
                    [(vbo, "3f 3f 2f", "in_pos", "in_normal", "in_uv")],
                    index_buffer=ibo,
                )
                tex = ctx.texture((image.shape[1], image.shape[0]), 4, image.tobytes())
                tex.build_mipmaps()
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                return MGLRenderable(
                    vao=vao,
                    vbo=vbo,
                    ibo=ibo,
                    prog=prog,
                    texture=tex,
                    mode=moderngl.TRIANGLES,
                    num_vertices=int(faces.size),
                )

        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(verts):
            colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
        else:
            colors = np.full((len(verts), 3), 0.75, dtype=np.float32)

        interleaved = np.hstack([verts, normals, colors]).astype(np.float32)
        vbo = ctx.buffer(interleaved.tobytes())
        ibo = ctx.buffer(faces.astype(np.int32).tobytes())
        prog = ctx.program(vertex_shader=_MESH_VERTEX_SHADER, fragment_shader=_MESH_FRAGMENT_SHADER)
        vao = ctx.vertex_array(
            prog,
            [(vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
            index_buffer=ibo,
        )
        return MGLRenderable(
            vao=vao,
            vbo=vbo,
            ibo=ibo,
            prog=prog,
            mode=moderngl.TRIANGLES,
            num_vertices=int(faces.size),
        )

    def to_moderngl_scene(self, ctx: moderngl.Context) -> Dict[str, MGLRenderable]:
        renderables: Dict[str, MGLRenderable] = {}
        Rg = np.asarray(self.R_scene, dtype=np.float64) if self.R_scene is not None else np.eye(3, dtype=np.float64)
        tg = np.asarray(self.t_scene, dtype=np.float64).reshape(-1) if self.t_scene is not None else np.zeros(3, dtype=np.float64)

        if self.bg_type == "floor_wall":
            if self.floor_corners is not None and self.floor_texture_path is not None:
                rf = self._make_quad_renderable(ctx, self.floor_corners, self.floor_texture_path, Rg, tg)
                if rf is not None:
                    renderables["floor"] = rf
            if self.wall_corners is not None and self.wall_texture_path is not None:
                rw = self._make_quad_renderable(ctx, self.wall_corners, self.wall_texture_path, Rg, tg)
                if rw is not None:
                    renderables["wall"] = rw
            return renderables

        if self.bg_type == "ply:splat":
            raise NotImplementedError("Gaussian Splatting is not supported in the ModernGL renderer yet.")

        if not isinstance(self.static_ply, trimesh.Geometry):
            return renderables

        geometries = self.static_ply.geometry.values() if isinstance(self.static_ply, trimesh.Scene) else [self.static_ply]
        for i, geom in enumerate(geometries):
            try:
                if isinstance(geom, trimesh.PointCloud):
                    pts = geom.vertices if hasattr(geom, "vertices") else geom.points
                    pts = _apply_scene_tf(np.asarray(pts, dtype=np.float32), Rg, tg)
                    if hasattr(geom, "colors") and geom.colors is not None:
                        cols = np.asarray(geom.colors[:, :3], dtype=np.float32)
                    else:
                        cols = np.full((len(pts), 3), 150.0, dtype=np.float32)
                    renderables[f"static_pcd_{i}"] = self._make_pointcloud_renderable(ctx, pts, cols, point_size=2.0)
                elif isinstance(geom, trimesh.Trimesh):
                    geom_t = geom.copy()
                    tf = np.eye(4, dtype=np.float64)
                    tf[:3, :3] = Rg
                    tf[:3, 3] = tg
                    geom_t.apply_transform(tf)
                    renderables[f"static_mesh_{i}"] = self._make_trimesh_renderable(ctx, geom_t)
            except Exception as exc:
                log(f"[{self.__class__.__name__}::to_moderngl_scene] Failed to add background geometry {i}: {exc}", "warning")
        return renderables

    @classmethod
    def from_capturestudio_dataset(cls, *args, **kwargs) -> "CapturestudioVirtualBackgroundModernGL":
        bg = super().from_capturestudio_dataset(*args, **kwargs)
        bg.__class__ = cls
        return bg

    @classmethod
    def from_object(cls, *args, **kwargs) -> "CapturestudioVirtualBackgroundModernGL":
        bg = super().from_object(*args, **kwargs)
        bg.__class__ = cls
        return bg


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualCamerasModernGL(CapturestudioVirtualCameras):
    def tick(self, renderables: Dict[str, MGLRenderable]) -> Optional[Tuple[int, int]]:
        super_out = super().tick()
        if super_out is None:
            return None
        t_prev, t_next = super_out

        if self.camera_show_virtual_frusta:
            prev_key = f"virt:{t_prev}"
            next_key = f"virt:{t_next}"
            if prev_key in renderables:
                vbo = renderables[prev_key].vbo
                data = np.frombuffer(vbo.read(), dtype=np.float32).reshape(-1, 6).copy()
                data[:, 3:] = [0.5, 0.5, 0.5]
                vbo.write(data.tobytes())
            if next_key in renderables:
                vbo = renderables[next_key].vbo
                data = np.frombuffer(vbo.read(), dtype=np.float32).reshape(-1, 6).copy()
                data[:, 3:] = [1.0, 0.0, 0.0]
                vbo.write(data.tobytes())
        return super_out

    def to_moderngl_scene(
        self,
        ctx: moderngl.Context,
        R_scene: Optional[np.ndarray] = None,
        t_scene: Optional[np.ndarray] = None,
    ) -> Dict[str, MGLRenderable]:
        renderables: Dict[str, MGLRenderable] = {}

        def _build_frustum(K: np.ndarray, c2w: np.ndarray, color: List[float]) -> MGLRenderable:
            c2w_v = np.eye(4, dtype=np.float64)
            c2w_v[:3, :3] = (R_scene @ c2w[:3, :3]) if R_scene is not None else c2w[:3, :3]
            c2w_v[:3, 3] = ((R_scene @ c2w[:3, 3]) + t_scene) if R_scene is not None else c2w[:3, 3]

            scale = float(self.frustum_scale)
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            w, h = self.image_size_hw[1], self.image_size_hw[0]
            pts_cam = np.array(
                [
                    [0, 0, 0],
                    [(0 - cx) * scale / fx, (0 - cy) * scale / fy, scale],
                    [(w - cx) * scale / fx, (0 - cy) * scale / fy, scale],
                    [(w - cx) * scale / fx, (h - cy) * scale / fy, scale],
                    [(0 - cx) * scale / fx, (h - cy) * scale / fy, scale],
                ],
                dtype=np.float32,
            )
            pts_world = (c2w_v[:3, :3] @ pts_cam.T).T + c2w_v[:3, 3]
            lines = np.array(
                [
                    pts_world[0], pts_world[1], pts_world[0], pts_world[2],
                    pts_world[0], pts_world[3], pts_world[0], pts_world[4],
                    pts_world[1], pts_world[2], pts_world[2], pts_world[3],
                    pts_world[3], pts_world[4], pts_world[4], pts_world[1],
                ],
                dtype=np.float32,
            )
            cols = np.tile(np.asarray(color, dtype=np.float32), (len(lines), 1))
            vbo_data = np.hstack([lines, cols]).astype(np.float32)
            prog = ctx.program(vertex_shader=_LINE_VERTEX_SHADER, fragment_shader=_LINE_FRAGMENT_SHADER)
            vbo = ctx.buffer(vbo_data.tobytes())
            vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_pos", "in_color")])
            return MGLRenderable(vao=vao, vbo=vbo, prog=prog, mode=moderngl.LINES, num_vertices=len(lines))

        if self.camera_show_virtual_frusta and self.virtual_extrinsics_c2ws is not None:
            self._compute_lookat_if_needed(R_scene, t_scene)
            self._roll_align_first_to_gt(R_scene, t_scene)
            for i in range(len(self.virtual_extrinsics_c2ws)):
                color = [1.0, 0.0, 0.0] if i == self._t_current else [0.5, 0.5, 0.5]
                c2w_prepared = self._prepare_virtual_c2w(i, R_scene, t_scene)
                renderables[f"virt:{i}"] = _build_frustum(self.virtual_intrinsics[i], c2w_prepared, color)

        self._R_scene = R_scene
        self._t_scene = t_scene
        return renderables

    @classmethod
    def from_capturestudio_dataset(cls, *args, **kwargs) -> "CapturestudioVirtualCamerasModernGL":
        c = super().from_capturestudio_dataset(*args, **kwargs)
        c.__class__ = cls
        return c


@dataclasses.dataclass(kw_only=True)
class CapturestudioVirtualDynamicForegroundModernGL(CapturestudioVirtualDynamicForeground):
    @functools.cached_property
    def label(self) -> str:
        return f"fg_{self.index:02d}"

    def tick(
        self,
        new_active_cam_idx: Optional[int] = None,
        target_w2c: Optional[np.ndarray] = None,
        renderables: Optional[Dict[str, MGLRenderable]] = None,
    ) -> None:
        super().tick(new_active_cam_idx=new_active_cam_idx)
        self.to_moderngl_scene(*self._last_vis_args, target_w2c=target_w2c, renderables=renderables)

    def to_moderngl_scene(
        self,
        ctx: moderngl.Context,
        R_scene: Optional[np.ndarray] = None,
        t_scene: Optional[np.ndarray] = None,
        target_w2c: Optional[np.ndarray] = None,
        renderables: Optional[Dict[str, MGLRenderable]] = None,
    ) -> Optional[Dict[str, MGLRenderable]]:
        self._last_vis_args = (ctx, R_scene, t_scene)
        image = next(self.image_generator)

        if isinstance(image, list):
            if self._blending_strategy == "swap":
                image = image[self._active_cam_idx]
            elif self._blending_strategy == "fuse":
                image = fuse_depth_maps(image)[self._active_cam_idx]
            elif self._blending_strategy.startswith("blend"):
                if target_w2c is None:
                    log(f"[{self.__class__.__name__}::to_moderngl_scene] target_w2c missing, falling back to active camera", "warning")
                    target_w2c = image[self._active_cam_idx].extrinsic_w2c
                closest_idx = list({max(0, self._active_cam_idx - 1), self._active_cam_idx})
                image = blend_point_cloud(
                    [image[i] for i in closest_idx],
                    target_w2c=target_w2c,
                    voxel_size_m=0.006,
                    angle_power=4.0,
                    refine_registration="deformation_pyramid",
                )
            elif self._blending_strategy.startswith("merge") and self._blending_strategy != "merge:naive":
                raise NotImplementedError(f"Unsupported blending strategy: {self._blending_strategy}")
            else:
                image = image[self._active_cam_idx]

        if isinstance(image, GSImage):
            raise NotImplementedError("Gaussian Splatting is not supported in the ModernGL renderer yet.")

        if isinstance(image, RGBDImage):
            image = image.unproject().open3d

        if not isinstance(image, o3d.geometry.PointCloud):
            raise NotImplementedError(f"Unexpected dynamic foreground type: {type(image)}")

        pts = np.asarray(image.points, dtype=np.float32)
        cols = np.asarray(image.colors, dtype=np.float32)
        pts = _apply_scene_tf(pts, R_scene, t_scene)
        label = f"{self.label}/pcd"
        vbo_data = np.hstack([pts, cols]).astype(np.float32)

        if renderables is None or label not in renderables:
            prog = ctx.program(vertex_shader=_PCD_VERTEX_SHADER, fragment_shader=_PCD_FRAGMENT_SHADER)
            vbo = ctx.buffer(vbo_data.tobytes())
            vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_pos", "in_color")])
            return {
                label: MGLRenderable(
                    vao=vao,
                    vbo=vbo,
                    prog=prog,
                    mode=moderngl.POINTS,
                    point_size=3.0,
                    num_vertices=len(pts),
                )
            }

        ren = renderables[label]
        if ren.num_vertices != len(pts):
            ren.vbo.orphan(vbo_data.nbytes)
            ren.vao = ctx.vertex_array(ren.prog, [(ren.vbo, "3f 3f", "in_pos", "in_color")])
            ren.num_vertices = len(pts)
        ren.vbo.write(vbo_data.tobytes())
        return {label: ren}

    @classmethod
    def from_capturestudio_dataset(cls, *args, **kwargs) -> "CapturestudioVirtualDynamicForegroundModernGL":
        fg = super().from_capturestudio_dataset(*args, **kwargs)
        fg.__class__ = cls
        return fg

    @classmethod
    def from_merged_ply_files(cls, *args, **kwargs) -> "CapturestudioVirtualDynamicForegroundModernGL":
        fg = super().from_merged_ply_files(*args, **kwargs)
        fg.__class__ = cls
        return fg


class CaptureStudioVirtualSceneModernGL(CapturestudioVirtualScene):
    def __init__(self, ctx: moderngl.Context, image_size_hw: Tuple[int, int], *args, **kwargs):
        self.ctx = ctx
        self.image_size_hw = image_size_hw
        self._all_renderables: Dict[str, MGLRenderable] = {}
        self._fbo: Optional[moderngl.Framebuffer] = None
        self._fbo_size: Optional[Tuple[int, int]] = None
        super().__init__(*args, **kwargs)

    def _get_fbo(self, target_size_hw: Tuple[int, int]) -> moderngl.Framebuffer:
        H, W = target_size_hw
        wanted = (W, H)
        if self._fbo is None or self._fbo_size != wanted:
            if self._fbo is not None:
                try:
                    self._fbo.release()
                except Exception:
                    pass
            self._fbo = self.ctx.simple_framebuffer(wanted, components=4)
            self._fbo_size = wanted
        return self._fbo

    def capture(self, target_size_hw: Tuple[int, int], show_background: bool = True) -> np.ndarray:
        H, W = target_size_hw
        fbo = self._get_fbo(target_size_hw)
        fbo.use()

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.line_width = 1.0
        fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        cam = self._cameras.current_camera
        K = np.asarray(cam["intrinsics"], dtype=np.float32)
        c2w = np.asarray(cam["extrinsic_c2w"], dtype=np.float32)
        view_mat = _get_view_matrix(c2w)
        proj_mat = _get_projection_matrix(K, W, H, near=float(cam["z_near"]), far=float(cam["z_far"]))
        mvp = proj_mat @ view_mat
        model = np.eye(4, dtype=np.float32)

        for key, ren in self._all_renderables.items():
            if not ren.visible:
                continue
            if key.startswith("background") and not show_background:
                continue

            if ren.texture is not None:
                ren.texture.use(location=0)
                if "u_tex" in ren.prog:
                    ren.prog["u_tex"].value = 0

            if "u_mvp" in ren.prog:
                ren.prog["u_mvp"].write(mvp.T.astype("f4").tobytes())
            if "u_model" in ren.prog:
                ren.prog["u_model"].write(model.T.astype("f4").tobytes())
            if "u_point_size" in ren.prog:
                ren.prog["u_point_size"].value = float(ren.point_size)

            if ren.mode == moderngl.LINES:
                self.ctx.line_width = float(ren.line_width)

            ren.vao.render(mode=ren.mode)

        self.ctx.finish()
        raw = fbo.read(components=3, alignment=1)
        image = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
        return np.flipud(image)

    def on_scene_updated(self, what_updated: str, set_not_unset: bool = True) -> None:
        if set_not_unset:
            kwargs = {}
            if what_updated != "background":
                kwargs = dict(R_scene=self._R_scene, t_scene=self._t_scene)
            if what_updated == "foregrounds" and self._cameras is not None:
                kwargs["target_w2c"] = np.linalg.inv(self._cameras.current_camera["extrinsic_c2w"])

            attr = getattr(self, f"_{what_updated}")
            if not isinstance(attr, list):
                attr = [attr]

            for i, attr_i in enumerate(attr):
                added = attr_i.to_moderngl_scene(self.ctx, **kwargs)
                if added:
                    for hk, hv in added.items():
                        if not hk.startswith(f"{what_updated}:"):
                            hk = f"{what_updated}:{i:02d}:{hk}"
                        self._all_renderables[hk] = hv
        else:
            for hk in list(self._all_renderables.keys()):
                if hk.startswith(f"{what_updated}:"):
                    self._all_renderables[hk].release()
                    del self._all_renderables[hk]

    def flush(self):
        self.ctx.finish()
        return self

    def tick(self):
        for _asset in ["background", "cameras", "foregrounds", "lights"]:
            attr = getattr(self, f"_{_asset}")
            if attr is None or not hasattr((attr if not isinstance(attr, list) else attr[0]), "tick"):
                continue
            if not isinstance(attr, list):
                attr = [attr]
            for i, attr_i in enumerate(attr):
                attr_i_renderables = {
                    hk.split(":")[-1]: hv
                    for hk, hv in self._all_renderables.items()
                    if hk.startswith(f"{_asset}:{i:02d}:")
                }
                kwargs = dict(renderables=attr_i_renderables)
                if _asset == "foregrounds" and self._cameras is not None:
                    kwargs["new_active_cam_idx"] = self._cameras.gt_cam_index
                    kwargs["target_w2c"] = np.linalg.inv(self._cameras.current_camera["extrinsic_c2w"])
                try:
                    attr_i.tick(**kwargs)
                except TypeError:
                    # background/lights may ignore the kwargs shape
                    attr_i.tick()
        self.flush()

    @classmethod
    def from_capturestudio_session(
        cls,
        ctx: moderngl.Context,
        dataset_raw: MultiSessionDataset,
        dataset_vis: Optional[Union[MultiSessionDataset, List[MultiSessionDataset], Path, List[Path]]] = None,
        wall_overshoot_m: float = 1.3,
        use_gs: bool = False,
        t_start: Union[int, List[int]] = 0,
        t_total: Union[int, List[int]] = -1,
        camera_orbit_type: Literal["interpolated", "audience"] = "audience",
        **kwargs,
    ) -> "CaptureStudioVirtualSceneModernGL":
        if dataset_vis is None:
            dataset_vis = dataset_raw
        if not isinstance(dataset_vis, list):
            dataset_vis = [dataset_vis]
        if not isinstance(t_start, list):
            t_start = [t_start] * len(dataset_vis)
        if not isinstance(t_total, list):
            t_total = [t_total] * len(t_start)

        floor_wall_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith("floor") or k.startswith("wall")}
        bg_kwargs = {k[3:]: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith("bg_")}
        fg_kwargs = {k[3:]: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith("fg_")}

        if "obj_path" in bg_kwargs:
            scene_background = CapturestudioVirtualBackgroundModernGL.from_object(**bg_kwargs)
        else:
            scene_background = CapturestudioVirtualBackgroundModernGL.from_capturestudio_dataset(
                dataset_raw,
                wall_overshoot_m=wall_overshoot_m,
                **floor_wall_kwargs,
            )

        scene_cameras = CapturestudioVirtualCamerasModernGL.from_capturestudio_dataset(
            dataset_raw,
            background=scene_background,
            camera_orbit_type=camera_orbit_type,
            t_total=max(t_total),
            **{k: kwargs.pop(k) for k in list(kwargs.keys()) if k.startswith("camera")},
        )

        if isinstance(dataset_vis[0], MultiSessionDataset):
            scene_foregrounds = [
                CapturestudioVirtualDynamicForegroundModernGL.from_capturestudio_dataset(
                    dataset_vis_i,
                    use_gs=use_gs,
                    t_start=t_start_i,
                    t_total=t_total_i,
                    index=i,
                    active_cam_idx=scene_cameras.gt_cam_index,
                    **fg_kwargs,
                )
                for i, (dataset_vis_i, t_start_i, t_total_i) in enumerate(zip(dataset_vis, t_start, t_total))
            ]
        elif isinstance(dataset_vis[0], Path):
            scene_foregrounds = [
                CapturestudioVirtualDynamicForegroundModernGL.from_merged_ply_files(
                    ply_root=dataset_vis_i,
                    ply_format="{ply_index:06d}.ply",
                    t_start=t_start_i,
                    t_total=t_total_i,
                    index=i,
                )
                for i, (dataset_vis_i, t_start_i, t_total_i) in enumerate(zip(dataset_vis, t_start, t_total))
            ]
        else:
            raise NotImplementedError

        return (
            cls(ctx=ctx, image_size_hw=dataset_raw.target_image_size_hw, **kwargs)
            .set_background(background=scene_background)
            .set_foregrounds(foregrounds=scene_foregrounds)
            .set_cameras(cameras=scene_cameras)
        ).flush()


class TeaserGeneratorModernGL(TeaserGenerator):
    def __init__(self, *args, **kwargs):
        self.ctx = moderngl.create_standalone_context(require=330)
        self._show_background = kwargs.pop("show_background", True)
        super().__init__(*args, **kwargs)

    def _init_scene(self):
        if self.t_total == -1:
            self.t_total = min(len(_) for _ in self.datasets_vis)
        self._scene = CaptureStudioVirtualSceneModernGL.from_capturestudio_session(
            ctx=self.ctx,
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
            **self._scene_kwargs,
        )

    def _rendering_loop(self) -> Path:
        video_writer_ = None
        for _ in tqdm(range(self.t_total), desc=f'Generating video file "{self.out_video_path}"'):
            rendered_img_ = self._scene.capture(self.render_config.image_size_hw, show_background=self._show_background)

            if self.debug:
                out_file = Path(self.out_video_path).name.split(".")[0] + f'_t0{str(datetime.timedelta(milliseconds=self.t_start * 1 / 30 * 1000))[:-3].replace(":", "-").replace(".", "-")}.png'
                cv2.imwrite(out_file, cv2.cvtColor(rendered_img_, cv2.COLOR_RGB2BGR))
                print(f"done: {out_file}")
                raise SystemExit(-1)

            if video_writer_ is None:
                video_writer_ = cv2.VideoWriter(
                    self.out_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (rendered_img_.shape[1], rendered_img_.shape[0]),
                )
            video_writer_.write(cv2.cvtColor(rendered_img_, cv2.COLOR_RGB2BGR))
            self._scene.tick()

        if video_writer_ is not None:
            video_writer_.release()
            log(f'[{self.__class__.__name__}::_rendering_loop] Video file released: "{self.out_video_path}"', "info")
        return self.out_video_path

    def run(self) -> Path:
        return self._rendering_loop()


if __name__ == "__main__":
    # @formatter:off
    DATA = [
        ("Cagliari_1_Perf_7", "Cagliari_1_Calib_6", 0, list(range(1, 8))),
    ]
    # @formatter:on
    DEBUG = False

    import time as time_
    import gc as gc_

    for SESSION_PERF, SESSION_CALIB, T_START, CAM_IDX in DATA:
        video_paths_ = {}
        for use_gs_ in [False]:
            visualizer_ = TeaserGeneratorModernGL(
                session_perf=SESSION_PERF if isinstance(SESSION_PERF, list) else SESSION_PERF.split("|"),
                session_calib=SESSION_CALIB,
                calib_method="MultiCamCalib",
                depth_source="bilateral_temporal",
                cam_idx_perf=CAM_IDX,
                cam_idx_raw=CAM_IDX,
                render_config=TeaserGeneratorRenderConfig.for_apr_may_2025(
                    use_gs=use_gs_,
                    image_size_hw=(1080, 1920),
                    camera_traverse_velocity=0.2,
                    camera_orbit_offset_m=0.6,
                ),
                t_start=T_START,
                t_total=1 if DEBUG else -1,
                show_background=True,
                camera_orbit_start_idx=T_START,
                fg_blending_strategy="swap",
                debug=DEBUG,
            )
            video_path_ = visualizer_.run()
            video_paths_[f"{'gs' if use_gs_ else 'pcd'}_path"] = video_path_

        del visualizer_
        gc_.collect()
        torch.cuda.empty_cache()
        time_.sleep(1.0)