from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Optional, Sequence, Union

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from reconstruction.primitive.pcd import RGBDImage


def _normalize_np(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def _normalize_torch(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1e-8)


def _rodrigues_torch(w: torch.Tensor) -> torch.Tensor:
    w = w.reshape(3)
    theta = torch.linalg.norm(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    if bool(theta.detach().cpu() < 1e-12):
        return eye
    k = w / theta
    K = torch.stack([
        torch.stack([torch.zeros_like(k[0]), -k[2], k[1]]),
        torch.stack([k[2], torch.zeros_like(k[0]), -k[0]]),
        torch.stack([-k[1], k[0], torch.zeros_like(k[0])]),
    ])
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def _logit_clip(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


@dataclass
class FastGCOState:
    A: np.ndarray
    b: np.ndarray
    r: np.ndarray
    t: np.ndarray
    uv: np.ndarray
    conf: np.ndarray
    n_nodes: int
    image_hw: tuple[int, int]


class FastGCOTemporalCache:
    def __init__(self) -> None:
        self._states: dict[Hashable, FastGCOState] = {}

    def get(self, key: Hashable) -> Optional[FastGCOState]:
        return self._states.get(key)

    def put(self, key: Hashable, state: FastGCOState) -> None:
        self._states[key] = state

    def clear(self) -> None:
        self._states.clear()


@dataclass
class DeformationGraph:
    node_pos: np.ndarray
    node_indices: np.ndarray
    edges: np.ndarray
    point_knn_idx: np.ndarray
    point_knn_w: np.ndarray
    node_knn_idx: np.ndarray
    node_knn_w: np.ndarray

    @classmethod
    def from_points(
            cls,
            points: np.ndarray,
            *,
            node_stride: int = 24,
            k_neighbors: int = 4,
            edge_neighbors: int = 6,
    ) -> "DeformationGraph":
        points = np.asarray(points, dtype=np.float64)
        node_indices = np.arange(0, len(points), max(1, int(node_stride)), dtype=np.int32)
        node_pos = points[node_indices]
        if len(node_pos) < 8:
            node_indices = np.arange(len(points), dtype=np.int32)
            node_pos = points.copy()

        tree = cKDTree(node_pos)

        def _query(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            d, idx = tree.query(x, k=min(int(k), len(node_pos)))
            if np.ndim(idx) == 1:
                idx = idx[:, None]
                d = d[:, None]
            return np.asarray(d, dtype=np.float64), np.asarray(idx, dtype=np.int32)

        def _weights(d: np.ndarray) -> np.ndarray:
            dmax = d[:, -1:] + 1e-8
            w = np.clip(1.0 - d / dmax, 0.0, None)
            return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)

        d_pts, idx_pts = _query(points, k_neighbors)
        d_nodes, idx_nodes = _query(node_pos, k_neighbors)
        _, idx_edges = _query(node_pos, edge_neighbors + 1)

        edges = set()
        for i in range(len(node_pos)):
            for j in idx_edges[i, 1:]:
                if i != int(j):
                    edges.add(tuple(sorted((int(i), int(j)))))
        edges_arr = np.asarray(sorted(edges), dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)

        return cls(
            node_pos=node_pos,
            node_indices=node_indices,
            edges=edges_arr,
            point_knn_idx=idx_pts,
            point_knn_w=_weights(d_pts),
            node_knn_idx=idx_nodes,
            node_knn_w=_weights(d_nodes),
        )


@dataclass
class TargetDepthMapGPU:
    depth_map_t: torch.Tensor
    dzdu_map_t: torch.Tensor
    dzdv_map_t: torch.Tensor
    intrinsic_t: torch.Tensor
    c2w_t: torch.Tensor
    image_hw: tuple[int, int]
    valid_uv: np.ndarray
    valid_points_world: np.ndarray
    valid_normals_world: np.ndarray

    @classmethod
    def from_rgbd(
            cls,
            rgbd: RGBDImage,
            *,
            device: torch.device,
            hole_scale: float = 2.0,
            smooth_sigma: float = 2.0,
    ) -> "TargetDepthMapGPU":
        depth = np.asarray(rgbd.depth, dtype=np.float32)
        base_valid = (depth > 0.0) & np.isfinite(depth) & np.asarray(rgbd.mask, dtype=bool)
        if not np.any(base_valid):
            raise ValueError("Target depth has no valid pixels")

        deep_hole = float(hole_scale * np.nanmax(depth[base_valid]))
        filled = np.where(base_valid, depth, deep_hole).astype(np.float32)

        eroded = cv2.erode(base_valid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        feather_region = ~eroded
        smooth = cv2.GaussianBlur(filled, (0, 0), smooth_sigma)
        filled[feather_region] = smooth[feather_region]

        dzdv, dzdu = np.gradient(filled)

        H, W = filled.shape
        depth_map_t = torch.from_numpy(filled)[None, None].to(device=device, dtype=torch.float32)
        dzdu_map_t = torch.from_numpy(dzdu.astype(np.float32))[None, None].to(device=device, dtype=torch.float32)
        dzdv_map_t = torch.from_numpy(dzdv.astype(np.float32))[None, None].to(device=device, dtype=torch.float32)

        pp = rgbd.unproject()
        _ = pp.normals
        valid_mask = np.asarray(pp.valid, dtype=bool).copy()
        flat_idx = np.flatnonzero(valid_mask.reshape(-1))
        valid_points_world = np.asarray(pp.points, dtype=np.float64).reshape(-1, 3)[flat_idx]
        valid_normals_world = np.asarray(pp.normals, dtype=np.float64).reshape(-1, 3)[flat_idx]
        vv = flat_idx // W
        uu = flat_idx % W
        valid_uv = np.stack([uu, vv], axis=1).astype(np.float64)

        return cls(
            depth_map_t=depth_map_t,
            dzdu_map_t=dzdu_map_t,
            dzdv_map_t=dzdv_map_t,
            intrinsic_t=torch.from_numpy(np.asarray(rgbd.intrinsic, dtype=np.float32)).to(device),
            c2w_t=torch.from_numpy(np.linalg.inv(np.asarray(rgbd.extrinsic_w2c, dtype=np.float32))).to(device),
            image_hw=(H, W),
            valid_uv=valid_uv,
            valid_points_world=valid_points_world,
            valid_normals_world=valid_normals_world,
        )

    def _sample_map(self, map_t: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        H, W = self.image_hw
        u = torch.clamp(uv[:, 0], 0.0, float(W - 1))
        v = torch.clamp(uv[:, 1], 0.0, float(H - 1))
        x = 2.0 * (u / max(W - 1, 1)) - 1.0
        y = 2.0 * (v / max(H - 1, 1)) - 1.0
        grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
        out = F.grid_sample(map_t, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out.view(-1)

    def sample_world(self, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._sample_map(self.depth_map_t, uv)
        dzdu = self._sample_map(self.dzdu_map_t, uv)
        dzdv = self._sample_map(self.dzdv_map_t, uv)

        fx = self.intrinsic_t[0, 0]
        fy = self.intrinsic_t[1, 1]
        cx = self.intrinsic_t[0, 2]
        cy = self.intrinsic_t[1, 2]

        u = uv[:, 0]
        v = uv[:, 1]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        p_cam = torch.stack([x, y, z], dim=1)

        R = self.c2w_t[:3, :3]
        t = self.c2w_t[:3, 3]
        p_w = (R @ p_cam.T).T + t

        du_cam = torch.stack([
            z / fx + (u - cx) * dzdu / fx,
            (v - cy) * dzdu / fy,
            dzdu,
        ], dim=1)
        dv_cam = torch.stack([
            (u - cx) * dzdv / fx,
            z / fy + (v - cy) * dzdv / fy,
            dzdv,
        ], dim=1)
        n_cam = torch.cross(du_cam, dv_cam, dim=1)
        n_w = _normalize_torch((R @ n_cam.T).T)
        return p_w, n_w


def _apply_embedded_deformation_torch(
        A: torch.Tensor,
        b: torch.Tensor,
        points: torch.Tensor,
        g_center: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        node_pos: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_w: torch.Tensor,
        use_global_rigid: bool,
) -> torch.Tensor:
    neigh_A = A[knn_idx]
    neigh_b = b[knn_idx]
    neigh_x = node_pos[knn_idx]
    x_local = points[:, None, :] - neigh_x
    y = torch.einsum("mkab,mkb->mka", neigh_A, x_local) + neigh_x + neigh_b
    y = torch.sum(knn_w[..., None] * y, dim=1)
    if not use_global_rigid:
        return y
    R = _rodrigues_torch(r)
    return (R @ (y - g_center).T).T + g_center + t


class FastGlobalCorrespondenceOptimizer:
    def __init__(
            self,
            source_rgbd: RGBDImage,
            target_rgbd: RGBDImage,
            *,
            node_stride: int = 24,
            graph_k: int = 4,
            edge_neighbors: int = 6,
            alpha_fit: float = 1.0,
            alpha_rigid: float = 200.0,
            alpha_smooth: float = 50.0,
            alpha_conf: float = 50.0,
            alpha_b_anchor: float = 00.0, # TODO -> 10.0
            alpha_global_anchor: float = 50.0,
            alpha_uv_anchor: float = 2.0,
            uv_window_px: float = 24.0, # TODO: 6.0
            use_global_rigid: bool = False,
            device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.device = device
        self.use_global_rigid = bool(use_global_rigid)

        src_pp = source_rgbd.unproject()
        _ = src_pp.normals
        src_valid = np.asarray(src_pp.valid, dtype=bool).copy()
        flat_idx = np.flatnonzero(src_valid.reshape(-1))
        self.src_points = np.asarray(src_pp.points, dtype=np.float64).reshape(-1, 3)[flat_idx]
        self.src_normals = np.asarray(src_pp.normals, dtype=np.float64).reshape(-1, 3)[flat_idx]

        self.graph = DeformationGraph.from_points(
            self.src_points,
            node_stride=node_stride,
            k_neighbors=graph_k,
            edge_neighbors=edge_neighbors,
        )
        self.node_normals = self.src_normals[self.graph.node_indices]
        self.g_center_np = self.graph.node_pos.mean(axis=0)
        bbox_min = self.src_points.min(axis=0)
        bbox_max = self.src_points.max(axis=0)
        self.scene_scale = float(np.linalg.norm(bbox_max - bbox_min) + 1e-8)

        self.target = TargetDepthMapGPU.from_rgbd(target_rgbd, device=self.device)
        self.target_tree = cKDTree(self.target.valid_points_world)

        self.alpha_fit = float(alpha_fit)
        self.alpha_rigid = float(alpha_rigid)
        self.alpha_smooth = float(alpha_smooth)
        self.alpha_conf = float(alpha_conf)
        self.alpha_b_anchor = float(alpha_b_anchor)
        self.alpha_global_anchor = float(alpha_global_anchor)
        self.alpha_uv_anchor = float(alpha_uv_anchor)
        self.uv_window_px = float(uv_window_px)

        self.src_points_t = torch.from_numpy(self.src_points.astype(np.float32)).to(self.device)
        self.graph_node_pos_t = torch.from_numpy(self.graph.node_pos.astype(np.float32)).to(self.device)
        self.node_normals_t = torch.from_numpy(self.node_normals.astype(np.float32)).to(self.device)
        self.g_center_t = torch.from_numpy(self.g_center_np.astype(np.float32)).to(self.device)
        self.point_knn_idx_t = torch.from_numpy(self.graph.point_knn_idx.astype(np.int64)).to(self.device)
        self.point_knn_w_t = torch.from_numpy(self.graph.point_knn_w.astype(np.float32)).to(self.device)
        self.node_knn_idx_t = torch.from_numpy(self.graph.node_knn_idx.astype(np.int64)).to(self.device)
        self.node_knn_w_t = torch.from_numpy(self.graph.node_knn_w.astype(np.float32)).to(self.device)
        self.edges_t = torch.from_numpy(self.graph.edges.astype(np.int64)).to(self.device) if len(self.graph.edges) else None
        self.scene_scale_t = torch.tensor(self.scene_scale, device=self.device, dtype=torch.float32)

    @property
    def n_nodes(self) -> int:
        return int(self.graph.node_pos.shape[0])

    def _project_world_to_uv(self, points_world: np.ndarray) -> np.ndarray:
        c2w = self.target.c2w_t.detach().cpu().numpy()
        w2c = np.linalg.inv(c2w)
        p_cam = (w2c[:3, :3] @ points_world.T).T + w2c[:3, 3]
        z = np.maximum(p_cam[:, 2], 1e-8)
        K = self.target.intrinsic_t.detach().cpu().numpy()
        u = K[0, 0] * (p_cam[:, 0] / z) + K[0, 2]
        v = K[1, 1] * (p_cam[:, 1] / z) + K[1, 2]
        uv = np.stack([u, v], axis=1)
        H, W = self.target.image_hw
        uv[:, 0] = np.clip(uv[:, 0], 0.0, W - 1.0)
        uv[:, 1] = np.clip(uv[:, 1], 0.0, H - 1.0)
        return uv.astype(np.float32)

    def _encode_conf(self, conf: np.ndarray) -> np.ndarray:
        return _logit_clip(conf)

    def _decode_conf(self, conf_raw_t: torch.Tensor) -> torch.Tensor:
        # keep a small floor so the system cannot completely turn off the fit and drift.
        return 0.1 + 0.9 * torch.sigmoid(conf_raw_t)

    def _state_to_init(self, warm_start: Optional[FastGCOState]):
        n = self.n_nodes
        uv_ref = self._project_world_to_uv(self.graph.node_pos)
        if warm_start is not None and warm_start.n_nodes == n and warm_start.image_hw == self.target.image_hw:
            A = warm_start.A.astype(np.float32, copy=True)
            b = warm_start.b.astype(np.float32, copy=True)
            r = warm_start.r.astype(np.float32, copy=True)
            t = warm_start.t.astype(np.float32, copy=True)
            conf = warm_start.conf.astype(np.float32, copy=True)
            uv_ref = np.clip(warm_start.uv.astype(np.float32, copy=True), [0.0, 0.0], [self.target.image_hw[1] - 1.0, self.target.image_hw[0] - 1.0])
        else:
            A = np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0)
            b = np.zeros((n, 3), dtype=np.float32)
            r = np.zeros(3, dtype=np.float32)
            t = np.zeros(3, dtype=np.float32)
            conf = np.ones(n, dtype=np.float32) * 0.95
        return A, b, r, t, uv_ref, conf

    def _decode_uv(self, duv_raw_t: torch.Tensor, uv_ref_t: torch.Tensor) -> torch.Tensor:
        duv = self.uv_window_px * torch.tanh(duv_raw_t)
        uv = uv_ref_t + duv
        H, W = self.target.image_hw
        u = torch.clamp(uv[:, 0], 0.0, float(W - 1))
        v = torch.clamp(uv[:, 1], 0.0, float(H - 1))
        return torch.stack([u, v], dim=1)

    def _compute_deformed_node_normals(self, A: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        local_A = torch.sum(self.node_knn_w_t[..., None, None] * A[self.node_knn_idx_t], dim=1)
        invT = torch.linalg.pinv(local_A).transpose(1, 2)
        n_local = torch.einsum("nab,nb->na", invT, self.node_normals_t)
        if self.use_global_rigid:
            R = _rodrigues_torch(r)
            n_local = (R @ n_local.T).T
        return _normalize_torch(n_local)

    def _loss(
            self,
            A_param: torch.nn.Parameter,
            b: torch.nn.Parameter,
            r: torch.nn.Parameter,
            t: torch.nn.Parameter,
            duv_raw: torch.nn.Parameter,
            conf_raw: torch.nn.Parameter,
            uv_ref_t: torch.Tensor,
            alpha_fit: float,
            alpha_rigid: float,
            alpha_smooth: float,
            alpha_conf: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eye = torch.eye(3, device=self.device, dtype=torch.float32)[None]
        A = eye + A_param
        uv = self._decode_uv(duv_raw, uv_ref_t)
        conf = self._decode_conf(conf_raw)

        nodes_def = _apply_embedded_deformation_torch(
            A, b, self.graph_node_pos_t, self.g_center_t, r, t,
            self.graph_node_pos_t, self.node_knn_idx_t, self.node_knn_w_t,
            self.use_global_rigid,
        )
        c_w, _ = self.target.sample_world(uv)

        # FIXED: Use torch.sum and divide by self.n_nodes uniformly across ALL terms
        # This preserves the relative scaling of the original SGP paper.
        fit = torch.sum((conf[:, None] * (nodes_def - c_w)) ** 2) / self.n_nodes

        a1, a2, a3 = A[:, :, 0], A[:, :, 1], A[:, :, 2]
        rigid_res = torch.stack([
            torch.sum(a1 * a2, dim=1),
            torch.sum(a1 * a3, dim=1),
            torch.sum(a2 * a3, dim=1),
            1.0 - torch.sum(a1 * a1, dim=1),
            1.0 - torch.sum(a2 * a2, dim=1),
            1.0 - torch.sum(a3 * a3, dim=1),
        ], dim=1)
        rigid = torch.sum(rigid_res ** 2) / self.n_nodes

        if self.edges_t is not None and self.edges_t.numel() > 0:
            i, j = self.edges_t[:, 0], self.edges_t[:, 1]
            smooth_res = (
                    torch.einsum("eab,eb->ea", A[i], self.graph_node_pos_t[j] - self.graph_node_pos_t[i])
                    + self.graph_node_pos_t[i] + b[i]
                    - (self.graph_node_pos_t[j] + b[j])
            )
            smooth = torch.sum(smooth_res ** 2) / self.n_nodes
        else:
            smooth = torch.zeros([], device=self.device, dtype=torch.float32)

        conf_pen = torch.sum((1.0 - conf ** 2) ** 2) / self.n_nodes
        b_anchor = torch.sum((b / self.scene_scale_t) ** 2) / self.n_nodes
        duv = uv - uv_ref_t
        uv_anchor = torch.sum((duv / max(self.uv_window_px, 1.0)) ** 2) / self.n_nodes

        if self.use_global_rigid:
            global_anchor = (torch.sum(r ** 2) + torch.sum((t / self.scene_scale_t) ** 2)) / self.n_nodes
        else:
            global_anchor = torch.zeros([], device=self.device, dtype=torch.float32)

        total = (
                alpha_fit * fit
                + alpha_rigid * rigid
                + alpha_smooth * smooth
                + alpha_conf * conf_pen
                + self.alpha_b_anchor * b_anchor
                + self.alpha_uv_anchor * uv_anchor
                + self.alpha_global_anchor * global_anchor
        )
        return total, {
            "fit": fit.detach(),
            "rigid": rigid.detach(),
            "smooth": smooth.detach(),
            "conf": conf_pen.detach(),
            "b_anchor": b_anchor.detach(),
            "uv_anchor": uv_anchor.detach(),
            "global_anchor": global_anchor.detach(),
        }

    def _loss2(
            self,
            A_param: torch.nn.Parameter,
            b: torch.nn.Parameter,
            r: torch.nn.Parameter,
            t: torch.nn.Parameter,
            duv_raw: torch.nn.Parameter,
            conf_raw: torch.nn.Parameter,
            uv_ref_t: torch.Tensor,
            alpha_fit: float,
            alpha_rigid: float,
            alpha_smooth: float,
            alpha_conf: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eye = torch.eye(3, device=self.device, dtype=torch.float32)[None]
        A = eye + A_param
        uv = self._decode_uv(duv_raw, uv_ref_t)
        conf = self._decode_conf(conf_raw)

        nodes_def = _apply_embedded_deformation_torch(
            A, b, self.graph_node_pos_t, self.g_center_t, r, t,
            self.graph_node_pos_t, self.node_knn_idx_t, self.node_knn_w_t,
            self.use_global_rigid,
        )
        c_w, _ = self.target.sample_world(uv)

        fit = torch.mean((conf[:, None] * (nodes_def - c_w)) ** 2)

        a1 = A[:, :, 0]
        a2 = A[:, :, 1]
        a3 = A[:, :, 2]
        rigid_res = torch.stack([
            torch.sum(a1 * a2, dim=1),
            torch.sum(a1 * a3, dim=1),
            torch.sum(a2 * a3, dim=1),
            1.0 - torch.sum(a1 * a1, dim=1),
            1.0 - torch.sum(a2 * a2, dim=1),
            1.0 - torch.sum(a3 * a3, dim=1),
        ], dim=1)
        rigid = torch.mean(rigid_res ** 2)

        if self.edges_t is not None and self.edges_t.numel() > 0:
            i = self.edges_t[:, 0]
            j = self.edges_t[:, 1]
            smooth_res = (
                    torch.einsum("eab,eb->ea", A[i], self.graph_node_pos_t[j] - self.graph_node_pos_t[i])
                    + self.graph_node_pos_t[i] + b[i]
                    - (self.graph_node_pos_t[j] + b[j])
            )
            smooth = torch.mean(smooth_res ** 2)
        else:
            smooth = torch.zeros([], device=self.device, dtype=torch.float32)

        conf_pen = torch.mean((1.0 - conf ** 2) ** 2)
        b_anchor = torch.mean((b / self.scene_scale_t) ** 2)
        duv = uv - uv_ref_t
        uv_anchor = torch.mean((duv / max(self.uv_window_px, 1.0)) ** 2)
        if self.use_global_rigid:
            global_anchor = torch.mean(r ** 2) + torch.mean((t / self.scene_scale_t) ** 2)
        else:
            global_anchor = torch.zeros([], device=self.device, dtype=torch.float32)

        total = (
                alpha_fit * fit
                + alpha_rigid * rigid
                + alpha_smooth * smooth
                + alpha_conf * conf_pen
                + self.alpha_b_anchor * b_anchor
                + self.alpha_uv_anchor * uv_anchor
                + self.alpha_global_anchor * global_anchor
        )
        return total, {
            "fit": fit.detach(),
            "rigid": rigid.detach(),
            "smooth": smooth.detach(),
            "conf": conf_pen.detach(),
            "b_anchor": b_anchor.detach(),
            "uv_anchor": uv_anchor.detach(),
            "global_anchor": global_anchor.detach(),
        }

    @torch.no_grad()
    def _refresh_correspondences(
            self,
            A_param: torch.nn.Parameter,
            b: torch.nn.Parameter,
            r: torch.nn.Parameter,
            t: torch.nn.Parameter,
            uv_ref_t: torch.Tensor,
            duv_raw: torch.nn.Parameter,
            conf_raw: torch.nn.Parameter,
            *,
            dist_thresh_m: float = 0.02,
            normal_dot_thresh: float = 0.6,
            max_jump_px: float = 6.0,
            max_jump_m: float = 0.03,
    ) -> None:
        eye = torch.eye(3, device=self.device, dtype=torch.float32)[None]
        A = eye + A_param
        nodes_def = _apply_embedded_deformation_torch(
            A, b, self.graph_node_pos_t, self.g_center_t, r, t,
            self.graph_node_pos_t, self.node_knn_idx_t, self.node_knn_w_t,
            self.use_global_rigid,
        )
        src_n = self._compute_deformed_node_normals(A, r)

        nodes_def_np = nodes_def.detach().cpu().numpy().astype(np.float64)
        src_n_np = src_n.detach().cpu().numpy().astype(np.float64)
        uv_prev_np = uv_ref_t.detach().cpu().numpy().astype(np.float64)

        dist, nn = self.target_tree.query(nodes_def_np, k=1)
        uv_nn = self.target.valid_uv[nn].astype(np.float64)
        tgt_n = self.target.valid_normals_world[nn]
        uv_jump = np.linalg.norm(uv_nn - uv_prev_np, axis=1)
        good = (
                (dist < dist_thresh_m)
                & (dist < max_jump_m)
                & (uv_jump < max_jump_px)
                & (np.sum(src_n_np * tgt_n, axis=1) > normal_dot_thresh)
        )

        uv_new = np.where(good[:, None], uv_nn, uv_prev_np).astype(np.float32)
        conf_new = np.where(good, 0.95, 0.2).astype(np.float32)

        uv_ref_t.data.copy_(torch.from_numpy(uv_new).to(self.device))
        duv_raw.data.zero_()
        conf_raw.data.copy_(torch.from_numpy(self._encode_conf(conf_new)).to(self.device))

    def optimize(
            self,
            *,
            warm_start: Optional[FastGCOState] = None,
            cache: Optional[FastGCOTemporalCache] = None,
            cache_key: Optional[Hashable] = None,
            steps_per_stage: Sequence[int] = (24, 16, 8),
            lrs: Sequence[float] = (2e-2, 1e-2, 5e-3),
            refresh_each_stage: bool = True,
            use_lbfgs_polish: bool = False,
            verbose: int = 0,
    ) -> dict[str, Any]:
        if warm_start is None and cache is not None and cache_key is not None:
            warm_start = cache.get(cache_key)

        A0, b0, r0, t0, uv_ref0, conf0 = self._state_to_init(warm_start)
        eye_np = np.eye(3, dtype=np.float32)[None]

        A_param = torch.nn.Parameter(torch.from_numpy(A0 - eye_np).to(self.device))
        b = torch.nn.Parameter(torch.from_numpy(b0).to(self.device))
        r = torch.nn.Parameter(torch.from_numpy(r0).to(self.device))
        t = torch.nn.Parameter(torch.from_numpy(t0).to(self.device))
        uv_ref_t = torch.from_numpy(uv_ref0).to(self.device)
        duv_raw = torch.nn.Parameter(torch.zeros_like(uv_ref_t))
        conf_raw = torch.nn.Parameter(torch.from_numpy(self._encode_conf(conf0)).to(self.device))

        stages = [
            (int(steps_per_stage[0]), float(lrs[0]), self.alpha_fit, self.alpha_rigid, self.alpha_smooth, self.alpha_conf),
            (int(steps_per_stage[1]), float(lrs[1]), self.alpha_fit, self.alpha_rigid * 0.5, self.alpha_smooth * 0.5, self.alpha_conf * 0.75),
            (int(steps_per_stage[2]), float(lrs[2]), self.alpha_fit, self.alpha_rigid * 0.25, self.alpha_smooth * 0.25, self.alpha_conf * 0.5),
        ]

        hist: list[float] = []
        for stage_idx, (steps, lr, afit, arigid, asmooth, aconf) in enumerate(stages):
            opt = torch.optim.Adam([A_param, b, r, t, duv_raw, conf_raw], lr=lr)
            for it in range(steps):
                opt.zero_grad(set_to_none=True)
                loss, parts = self._loss(A_param, b, r, t, duv_raw, conf_raw, uv_ref_t, afit, arigid, asmooth, aconf)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([A_param, b, r, t, duv_raw, conf_raw], max_norm=10.0)
                opt.step()

                if verbose and (it == 0 or (it + 1) % 10 == 0 or it + 1 == steps):
                    parts_f = {k: float(v.detach().cpu()) for k, v in parts.items()}
                    print(
                        f"[FastGCOStable][stage {stage_idx}] iter {it + 1}/{steps} "
                        f"loss={float(loss.detach().cpu()):.6f} fit={parts_f['fit']:.6f} "
                        f"rigid={parts_f['rigid']:.6f} smooth={parts_f['smooth']:.6f} "
                        f"conf={parts_f['conf']:.6f} b={parts_f['b_anchor']:.6f} uv={parts_f['uv_anchor']:.6f}"
                    )
                hist.append(float(loss.detach().cpu()))

            if refresh_each_stage and stage_idx < len(stages) - 1:
                self._refresh_correspondences(A_param, b, r, t, uv_ref_t, duv_raw, conf_raw)

        if use_lbfgs_polish:
            opt = torch.optim.LBFGS([A_param, b, r, t, duv_raw, conf_raw], lr=0.5, max_iter=8, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad(set_to_none=True)
                loss, _ = self._loss(A_param, b, r, t, duv_raw, conf_raw, uv_ref_t,
                                     self.alpha_fit, self.alpha_rigid * 0.25,
                                     self.alpha_smooth * 0.25, self.alpha_conf * 0.5)
                loss.backward()
                return loss

            opt.step(closure)

        with torch.no_grad():
            eye = torch.eye(3, device=self.device, dtype=torch.float32)[None]
            A = eye + A_param
            uv_t = self._decode_uv(duv_raw, uv_ref_t)
            conf_t = self._decode_conf(conf_raw)
            deformed_points_t = _apply_embedded_deformation_torch(
                A, b, self.src_points_t, self.g_center_t, r, t,
                self.graph_node_pos_t, self.point_knn_idx_t, self.point_knn_w_t,
                self.use_global_rigid,
            )
            nodes_def_t = _apply_embedded_deformation_torch(
                A, b, self.graph_node_pos_t, self.g_center_t, r, t,
                self.graph_node_pos_t, self.node_knn_idx_t, self.node_knn_w_t,
                self.use_global_rigid,
            )

            state = FastGCOState(
                A=A.detach().cpu().numpy().astype(np.float32),
                b=b.detach().cpu().numpy().astype(np.float32),
                r=r.detach().cpu().numpy().astype(np.float32),
                t=t.detach().cpu().numpy().astype(np.float32),
                uv=uv_t.detach().cpu().numpy().astype(np.float32),
                conf=conf_t.detach().cpu().numpy().astype(np.float32),
                n_nodes=self.n_nodes,
                image_hw=self.target.image_hw,
            )

            if cache is not None and cache_key is not None:
                cache.put(cache_key, state)

            return {
                "deformed_points": deformed_points_t.detach().cpu().numpy().astype(np.float32),
                "graph_nodes_deformed": nodes_def_t.detach().cpu().numpy().astype(np.float32),
                "graph_uv": uv_t.detach().cpu().numpy().astype(np.float32),
                "graph_conf": conf_t.detach().cpu().numpy().astype(np.float32),
                "warm_start": state,
                "loss_history": hist,
                "device": str(self.device),
            }


def warp_view_to_reference_fast(
        view: RGBDImage,
        ref_view: RGBDImage,
        *,
        warm_start: Optional[FastGCOState] = None,
        cache: Optional[FastGCOTemporalCache] = None,
        cache_key: Optional[Hashable] = None,
        node_stride: int = 32,
        graph_k: int = 4,
        edge_neighbors: int = 6,
        steps_per_stage: Sequence[int] = (24, 16, 8),
        lrs: Sequence[float] = (2e-2, 1e-2, 5e-3),
        use_global_rigid: bool = False,
        uv_window_px: float = 6.0,
        device: Optional[Union[str, torch.device]] = None,
        verbose: int = 0,
) -> tuple[o3d.geometry.PointCloud, dict[str, Any]]:
    opt = FastGlobalCorrespondenceOptimizer(
        source_rgbd=view,
        target_rgbd=ref_view,
        node_stride=node_stride,
        graph_k=graph_k,
        edge_neighbors=edge_neighbors,
        uv_window_px=uv_window_px,
        use_global_rigid=use_global_rigid,
        device=device,
    )
    result = opt.optimize(
        warm_start=warm_start,
        cache=cache,
        cache_key=cache_key,
        steps_per_stage=steps_per_stage,
        lrs=lrs,
        verbose=verbose,
    )

    src_pp = view.unproject()
    _ = src_pp.open3d
    src_valid = np.asarray(src_pp.valid, dtype=bool).copy()
    flat_idx = np.flatnonzero(src_valid.reshape(-1))
    colors = np.asarray(src_pp.colors, dtype=np.float32).reshape(-1, 3)[flat_idx]
    points = np.asarray(result["deformed_points"], dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print('mean', result["graph_conf"].mean())
    return pcd, result
