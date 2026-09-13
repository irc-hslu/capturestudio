from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from utils.misc import log


class VideoColorMatcher:
    class ColorNet(torch.nn.Module):
        def __init__(self, cfg: Dict[str, float]):
            super().__init__()
            self.cfg = dict(cfg)

            ev_min = float(cfg["ev_min"])
            ev_max = float(cfg["ev_max"])
            ev_init = float(cfg["ev_init"])
            ev_p = np.clip((ev_init - ev_min) / (ev_max - ev_min), 1e-4, 1.0 - 1e-4)
            ev_raw_init = np.log(ev_p / (1.0 - ev_p))

            self.ev_raw = torch.nn.Parameter(torch.tensor(float(ev_raw_init), dtype=torch.float32))
            self.wb_raw = torch.nn.Parameter(torch.zeros(2))

            # Starts near zero, but can only become positive.
            self.tone_raw = torch.nn.Parameter(torch.full((int(cfg["tone_knots"]),), -4.0))

            hidden = int(cfg["hidden"])
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(6, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, 3),
            )

            torch.nn.init.zeros_(self.mlp[-1].weight)
            torch.nn.init.zeros_(self.mlp[-1].bias)

        def forward(self, x_bgr: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            cfg = self.cfg
            n_knots = int(cfg["tone_knots"])

            ev = float(cfg["ev_min"]) + (float(cfg["ev_max"]) - float(cfg["ev_min"])) * torch.sigmoid(self.ev_raw)
            wb_br = float(cfg["max_wb"]) * torch.tanh(self.wb_raw)
            wb = torch.stack([wb_br[0], torch.zeros_like(wb_br[0]), wb_br[1]])

            x = VideoColorMatcher.srgb_to_linear(torch.clamp(x_bgr, 0.0, 1.0))
            x = x * torch.pow(2.0, ev)
            x = x * torch.pow(2.0, wb).view(*([1] * (x.ndim - 1)), 3)

            lum_w = torch.tensor([0.114, 0.587, 0.299], device=x.device, dtype=x.dtype)
            lum = torch.sum(x * lum_w.view(*([1] * (x.ndim - 1)), 3), dim=-1)

            tone_knots = float(cfg["max_tone_ev"]) * torch.sigmoid(self.tone_raw)
            pos = torch.clamp(lum, 0.0, 1.0) * float(n_knots - 1)
            i0 = torch.floor(pos).long().clamp(0, n_knots - 1)
            i1 = torch.clamp(i0 + 1, 0, n_knots - 1)
            t = (pos - i0.float()).unsqueeze(-1)

            tone = tone_knots[i0].unsqueeze(-1) * (1.0 - t) + tone_knots[i1].unsqueeze(-1) * t
            x = x * torch.pow(2.0, tone)

            y = VideoColorMatcher.linear_to_srgb(x)

            y_lum = torch.sum(y * lum_w.view(*([1] * (y.ndim - 1)), 3), dim=-1, keepdim=True)
            residual_in = torch.cat([y, y_lum, y[..., 0:1] - y_lum, y[..., 2:3] - y_lum], dim=-1)
            residual = float(cfg["max_residual"]) * torch.tanh(self.mlp(residual_in))
            y = torch.clamp(y + residual, 0.0, 1.0)

            d1 = tone_knots[1:] - tone_knots[:-1]
            d2 = d1[1:] - d1[:-1]

            aux = {
                "ev": ev,
                "wb": wb,
                "tone": tone_knots,
                "tone_smooth": d2.pow(2).mean(),
                "tone_energy": tone_knots.pow(2).mean(),
                "residual_energy": residual.pow(2).mean(),
            }

            return y, aux

    def __init__(
            self,
            recording_dir: Union[str, Path],
            hosts: Dict[int, Dict],
            device: Optional[str] = None,
            sample_frames: int = 30,
            fg_pixels: int = 8000,
            bg_pixels: int = 3000,
            max_pixels: int = 200_000,
            steps: int = 5000,
            batch: int = 16384,
            lr: float = 1e-3,
            seed: int = 42,
    ):
        self.recording_dir = Path(recording_dir)
        self.orbbec_dir = self.recording_dir / "orbbec"
        self.hosts = hosts
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)

        self.sample_frames = int(sample_frames)
        self.fg_pixels = int(fg_pixels)
        self.bg_pixels = int(bg_pixels)
        self.max_pixels = int(max_pixels)
        self.steps = int(steps)
        self.batch = int(batch)
        self.lr = float(lr)

        self.cfg = {
            "mask_threshold": 16,
            "mask_erode": 3,
            "min_pixels": 512,
            "min_value": 4,
            "max_value": 250,

            "ev_min": 0.0,
            "ev_max": 5.0,
            "ev_init": 1.5,

            "max_wb": 0.16,
            "tone_knots": 12,
            "max_tone_ev": 2.0,
            "hidden": 32,
            "max_residual": 0.08,

            "fg_target_lift_ev": 0.70,
            "bg_target_lift_ev": 0.60,

            # Manual visual lift applied at inference.
            "look_ev": 0.75,
            "look_by_cam": {
                3: {"fg": 2.50, "bg": 2.30},
                5: {"fg": 0.50, "bg": -0.50},
            },

            "fg_weight": 2.8,
            "bg_luma_weight": 1.2,

            "ev_reg": 0.0,
            "wb_reg": 0.01,
            "tone_smooth_reg": 0.04,
            "tone_energy_reg": 0.001,
            "residual_reg": 0.03,
            "clip_reg": 0.04,

            "fg_under_weight": 3.0,
            "bg_under_weight": 1.5,
        }

        master_hosts = [int(h) for h, spec in hosts.items() if bool(spec.get("is_master", False))]
        if len(master_hosts) != 1:
            raise ValueError(f"Expected exactly one master host, got {master_hosts}")

        self.master_host = master_hosts[0]
        self.master_cams = [int(c) for c in hosts[self.master_host]["cams"]]
        self.slave_cams = [
            int(c)
            for host_id, spec in sorted(hosts.items())
            if int(host_id) != self.master_host
            for c in spec["cams"]
        ]

        self.frames_by_cam: Dict[int, List[Path]] = {}
        for cam in sorted(set(self.master_cams + self.slave_cams)):
            color_dir = self.orbbec_dir / f"cam{cam:02d}" / "color"
            frames = []
            for ext in ("jpg", "jpeg", "png"):
                frames.extend(color_dir.glob(f"*.{ext}"))
            frames = sorted(frames, key=lambda p: int(p.stem))
            if not frames:
                raise FileNotFoundError(f"No color frames found in {color_dir}")
            self.frames_by_cam[cam] = frames

        self.total_frames = min(len(v) for v in self.frames_by_cam.values())
        self.frame_indices = sorted(
            set(
                np.linspace(
                    0,
                    self.total_frames - 1,
                    min(self.sample_frames, self.total_frames),
                ).round().astype(np.int64).tolist()
            )
        )

        self.model_dir = self.recording_dir / "color_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        log(
            f"[{self.__class__.__name__}] device={self.device}, "
            f"master_cams={self.master_cams}, slave_cams={self.slave_cams}, "
            f"frames={self.frame_indices}",
            "debug",
        )

    @staticmethod
    def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x <= 0.04045,
            x / 12.92,
            torch.pow((x + 0.055) / 1.055, 2.4),
        )

    @staticmethod
    def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        return torch.where(
            x <= 0.0031308,
            12.92 * x,
            1.055 * torch.pow(torch.clamp(x, min=1e-8), 1.0 / 2.4) - 0.055,
        )

    def apply_spatial_look(self, y: torch.Tensor, mask: Optional[np.ndarray], cam: int, cfg: Dict) -> torch.Tensor:
        look_cfg = cfg.get("look_by_cam", {})
        cam_cfg = look_cfg.get(int(cam), look_cfg.get(str(int(cam)), {"fg": 0.0, "bg": 0.0}))

        fg_ev = float(cam_cfg.get("fg", 0.0))
        bg_ev = float(cam_cfg.get("bg", 0.0))

        if abs(fg_ev) < 1e-8 and abs(bg_ev) < 1e-8:
            return y

        if mask is None:
            ev = torch.full((len(y), 1), fg_ev, device=y.device, dtype=y.dtype)
        else:
            m = cv2.resize(
                mask.astype(np.float32),
                (1, len(y)),
                interpolation=cv2.INTER_LINEAR,
            ).reshape(-1, 1)

            m = torch.from_numpy(m).to(y.device, dtype=y.dtype)
            ev = bg_ev * (1.0 - m) + fg_ev * m

        y_lin = self.srgb_to_linear(torch.clamp(y, 0.0, 1.0))
        y_lin = y_lin * torch.pow(torch.tensor(2.0, device=y.device, dtype=y.dtype), ev)
        return self.linear_to_srgb(y_lin)

    def _pixels(self, cam: int, frame_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        img_path = self.frames_by_cam[int(cam)][int(frame_idx)]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return None

        mask = None
        mask_dir = self.orbbec_dir / f"cam{int(cam):02d}" / "mask"
        for ext in ("jpg", "jpeg", "png"):
            p = mask_dir / f"{img_path.stem}.{ext}"
            if p.is_file():
                mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                break

        if mask is None:
            return None

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = mask > int(self.cfg["mask_threshold"])
        k = int(self.cfg["mask_erode"])

        if k > 0:
            kernel = np.ones((k, k), dtype=np.uint8)
            fg_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0

            kernel = np.ones((max(3, 2 * k + 1), max(3, 2 * k + 1)), dtype=np.uint8)
            bg_mask = ~(cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0)
        else:
            fg_mask = mask
            bg_mask = ~mask

        fg_y, fg_x = np.where(fg_mask)
        bg_y, bg_x = np.where(bg_mask)

        if len(fg_x) < int(self.cfg["min_pixels"]) or len(bg_x) < int(self.cfg["min_pixels"]):
            return None

        fg = img[fg_y, fg_x]
        bg = img[bg_y, bg_x]

        lo = int(self.cfg["min_value"])
        hi = int(self.cfg["max_value"])

        fg = fg[np.all((fg >= lo) & (fg <= hi), axis=1)]
        bg = bg[np.all((bg >= lo) & (bg <= hi), axis=1)]

        if len(fg) < int(self.cfg["min_pixels"]) or len(bg) < int(self.cfg["min_pixels"]):
            return None

        if len(fg) > self.fg_pixels:
            fg = fg[self.rng.choice(len(fg), self.fg_pixels, replace=False)]

        if len(bg) > self.bg_pixels:
            bg = bg[self.rng.choice(len(bg), self.bg_pixels, replace=False)]

        return fg.astype(np.uint8), bg.astype(np.uint8)

    def fit_camera(self, cam: int, save: bool = True) -> Dict[str, object]:
        cam = int(cam)

        if cam not in self.slave_cams:
            raise ValueError(f"cam{cam:02d} is not a slave camera")

        slave_fg, slave_bg, master_fg, master_bg = [], [], [], []

        for frame_idx in tqdm(self.frame_indices, desc=f"[{self.__class__.__name__}] cam{cam:02d} samples"):
            s = self._pixels(cam, frame_idx)
            if s is None:
                continue

            frame_master_fg = []
            frame_master_bg = []

            for master_cam in self.master_cams:
                m = self._pixels(master_cam, frame_idx)
                if m is None:
                    continue
                frame_master_fg.append(m[0])
                frame_master_bg.append(m[1])

            if not frame_master_fg:
                continue

            slave_fg.append(s[0])
            slave_bg.append(s[1])
            master_fg.append(np.concatenate(frame_master_fg, axis=0))
            master_bg.append(np.concatenate(frame_master_bg, axis=0))

            log(
                f"[samples] cam{cam:02d} frame={frame_idx:06d} "
                f"fg={len(s[0])}/{sum(len(x) for x in frame_master_fg)} "
                f"bg={len(s[1])}/{sum(len(x) for x in frame_master_bg)}",
                "debug",
            )

        if not slave_fg:
            raise RuntimeError(f"No samples collected for cam{cam:02d}")

        sx_fg = np.concatenate(slave_fg, axis=0).astype(np.uint8)
        sx_bg = np.concatenate(slave_bg, axis=0).astype(np.uint8)
        sy_fg = np.concatenate(master_fg, axis=0).astype(np.uint8)
        sy_bg = np.concatenate(master_bg, axis=0).astype(np.uint8)

        if len(sx_fg) > self.max_pixels:
            sx_fg = sx_fg[self.rng.choice(len(sx_fg), self.max_pixels, replace=False)]
        if len(sy_fg) > self.max_pixels:
            sy_fg = sy_fg[self.rng.choice(len(sy_fg), self.max_pixels, replace=False)]
        if len(sx_bg) > self.max_pixels:
            sx_bg = sx_bg[self.rng.choice(len(sx_bg), self.max_pixels, replace=False)]
        if len(sy_bg) > self.max_pixels:
            sy_bg = sy_bg[self.rng.choice(len(sy_bg), self.max_pixels, replace=False)]

        log(
            f"[samples] cam{cam:02d} "
            f"slave_fg_mean={sx_fg.mean(axis=0)} master_fg_mean={sy_fg.mean(axis=0)} "
            f"slave_bg_mean={sx_bg.mean(axis=0)} master_bg_mean={sy_bg.mean(axis=0)}",
            "debug",
        )

        x_fg = torch.from_numpy(sx_fg.astype(np.float32) / 255.0).to(self.device)
        y_fg = torch.from_numpy(sy_fg.astype(np.float32) / 255.0).to(self.device)
        x_bg = torch.from_numpy(sx_bg.astype(np.float32) / 255.0).to(self.device)
        y_bg = torch.from_numpy(sy_bg.astype(np.float32) / 255.0).to(self.device)

        fg_lift = float(self.cfg["fg_target_lift_ev"])
        bg_lift = float(self.cfg["bg_target_lift_ev"])

        if fg_lift != 0.0:
            y_fg = self.linear_to_srgb(
                self.srgb_to_linear(y_fg) * torch.pow(torch.tensor(2.0, device=self.device), fg_lift)
            ).detach()

        if bg_lift != 0.0:
            y_bg = self.linear_to_srgb(
                self.srgb_to_linear(y_bg) * torch.pow(torch.tensor(2.0, device=self.device), bg_lift)
            ).detach()

        model = self.ColorNet(self.cfg).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-5)

        q = torch.tensor([0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95], device=self.device)
        lum_w = torch.tensor([0.114, 0.587, 0.299], device=self.device).view(1, 3)

        for step in range(self.steps):
            ix_fg = torch.randint(0, len(x_fg), (min(self.batch, len(x_fg)),), device=self.device)
            iy_fg = torch.randint(0, len(y_fg), (min(self.batch, len(y_fg)),), device=self.device)
            ix_bg = torch.randint(0, len(x_bg), (min(self.batch, len(x_bg)),), device=self.device)
            iy_bg = torch.randint(0, len(y_bg), (min(self.batch, len(y_bg)),), device=self.device)

            xb_fg = x_fg[ix_fg]
            yb_fg = y_fg[iy_fg]
            xb_bg = x_bg[ix_bg]
            yb_bg = y_bg[iy_bg]

            pred_fg, aux = model(xb_fg)
            pred_bg, _ = model(xb_bg)

            pred_fg_l = (pred_fg * lum_w).sum(dim=-1)
            yb_fg_l = (yb_fg * lum_w).sum(dim=-1)

            fg_under = torch.relu(yb_fg_l.mean() - pred_fg_l.mean()).pow(2)

            fg_loss = (
                    0.70 * torch.sqrt((torch.quantile(pred_fg, q, dim=0) - torch.quantile(yb_fg, q, dim=0)) ** 2 + 1e-6).mean() +
                    0.80 * torch.sqrt((torch.quantile(pred_fg_l, q, dim=0) - torch.quantile(yb_fg_l, q, dim=0)) ** 2 + 1e-6).mean() +
                    0.25 * torch.sqrt((pred_fg.mean(dim=0) - yb_fg.mean(dim=0)) ** 2 + 1e-6).mean() +
                    0.20 * torch.sqrt((pred_fg.std(dim=0) - yb_fg.std(dim=0)) ** 2 + 1e-6).mean()
            )

            pred_bg_l = (pred_bg * lum_w).sum(dim=-1)
            xb_bg_l = (xb_bg * lum_w).sum(dim=-1)
            yb_bg_l = (yb_bg * lum_w).sum(dim=-1)

            bg_luma_loss = torch.sqrt(
                (torch.quantile(pred_bg_l, q, dim=0) - torch.quantile(yb_bg_l, q, dim=0)) ** 2 + 1e-6
            ).mean()

            bg_under = torch.relu(yb_bg_l.mean() - pred_bg_l.mean()).pow(2)

            reg = (
                    float(self.cfg["ev_reg"]) * (aux["ev"] - float(self.cfg["ev_init"])).pow(2) +
                    float(self.cfg["wb_reg"]) * aux["wb"].pow(2).mean() +
                    float(self.cfg["tone_smooth_reg"]) * aux["tone_smooth"] +
                    float(self.cfg["tone_energy_reg"]) * aux["tone_energy"] +
                    float(self.cfg["residual_reg"]) * aux["residual_energy"]
            )

            clip = (
                    torch.relu(pred_fg - 0.995).pow(2).mean() +
                    torch.relu(0.010 - pred_fg).pow(2).mean() +
                    torch.relu(pred_bg - 0.995).pow(2).mean() +
                    torch.relu(0.010 - pred_bg).pow(2).mean()
            )

            loss = (
                    float(self.cfg["fg_weight"]) * fg_loss +
                    float(self.cfg["bg_luma_weight"]) * bg_luma_loss +
                    float(self.cfg["fg_under_weight"]) * fg_under +
                    float(self.cfg["bg_under_weight"]) * bg_under +
                    reg +
                    float(self.cfg["clip_reg"]) * clip
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 100 == 0 or step == self.steps - 1:
                with torch.no_grad():
                    pred_eval, aux_eval = model(x_fg[:min(20_000, len(x_fg))])
                    target_eval = y_fg[torch.randint(0, len(y_fg), (len(pred_eval),), device=self.device)]
                    q_mae = torch.abs(
                        torch.quantile(pred_eval, q, dim=0) - torch.quantile(target_eval, q, dim=0)
                    ).mean().item() * 255.0
                    pred_mean = (pred_eval.mean(dim=0) * 255.0).detach().cpu().numpy()
                    target_mean = (target_eval.mean(dim=0) * 255.0).detach().cpu().numpy()

                    log(
                        f"[fit cam{cam:02d}] step={step:04d} "
                        f"loss={loss.item():.5f} fg={fg_loss.item():.5f} "
                        f"bg={bg_luma_loss.item():.5f} "
                        f"fg_under={fg_under.item():.5f} bg_under={bg_under.item():.5f} "
                        f"q_mae={q_mae:.2f} ev={aux_eval['ev'].item():+.3f} "
                        f"pred_mean={pred_mean} target_mean={target_mean} "
                        f"wb={aux_eval['wb'].detach().cpu().numpy()} "
                        f"tone=({aux_eval['tone'].min().item():+.3f},{aux_eval['tone'].max().item():+.3f}) "
                        f"res={aux_eval['residual_energy'].item():.6f}",
                        "debug",
                    )

        ckpt = {
            "cam": cam,
            "cfg": dict(self.cfg),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }

        if save:
            path = self.model_dir / f"cam{cam:02d}.pt"
            torch.save(ckpt, path)
            log(f"[save] {path}", "debug")

        return ckpt

    def fit_all(self, save: bool = True) -> Dict[int, Dict[str, object]]:
        out = {}
        for cam in self.slave_cams:
            out[cam] = self.fit_camera(cam, save=save)
        return out

    def apply(
            self,
            img_bgr: np.ndarray,
            ckpt: Dict[str, object],
            cam: int,
            mask: Optional[np.ndarray] = None,
            chunk: int = 1_000_000,
    ) -> np.ndarray:
        assert img_bgr.dtype == np.uint8
        assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3

        model = self.ColorNet(ckpt["cfg"]).to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        h, w = img_bgr.shape[:2]
        flat = img_bgr.reshape(-1, 3).astype(np.float32) / 255.0

        if mask is not None:
            mask_flat = mask.reshape(-1).astype(np.float32)
        else:
            mask_flat = None

        out = []

        with torch.inference_mode():
            for i in range(0, len(flat), int(chunk)):
                x = torch.from_numpy(flat[i:i + int(chunk)]).to(self.device)
                y, _ = model(x)

                if mask_flat is None:
                    mask_chunk = None
                else:
                    mask_chunk = mask_flat[i:i + int(chunk)]

                y = self.apply_spatial_look(y, mask_chunk, cam, ckpt["cfg"])
                out.append(torch.clamp(torch.round(y * 255.0), 0, 255).byte().cpu().numpy())

        return np.concatenate(out, axis=0).reshape(h, w, 3)

    def read_mask_for_frame(self, cam: int, img_path: Path) -> Optional[np.ndarray]:
        mask_dir = self.orbbec_dir / f"cam{int(cam):02d}" / "mask"

        mask = None
        for ext in ("jpg", "jpeg", "png"):
            p = mask_dir / f"{img_path.stem}.{ext}"
            if p.is_file():
                mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                break

        if mask is None:
            return None

        mask = mask > int(self.cfg["mask_threshold"])

        k = max(3, int(self.cfg["mask_erode"]) * 2 + 1)
        kernel = np.ones((k, k), dtype=np.uint8)

        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)

        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask = np.clip(mask, 0.0, 1.0)

        return mask

    def apply_look_ev(self, y: torch.Tensor, ev: float) -> torch.Tensor:
        if abs(float(ev)) < 1e-8:
            return y

        y_lin = self.srgb_to_linear(torch.clamp(y, 0.0, 1.0))
        y_lin = y_lin * torch.pow(torch.tensor(2.0, device=y.device, dtype=y.dtype), float(ev))
        return self.linear_to_srgb(y_lin)

    def apply_file(self, img_bgr: np.ndarray, cam: int, img_path: Optional[Path] = None) -> np.ndarray:
        path = self.model_dir / f"cam{int(cam):02d}.pt"
        if not path.is_file():
            return img_bgr

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")

        ckpt.setdefault("cfg", {})
        ckpt["cfg"]["look_by_cam"] = self.cfg.get("look_by_cam", {})

        mask = None
        if img_path is not None:
            mask = self.read_mask_for_frame(cam, img_path)
            if mask is not None and mask.shape[:2] != img_bgr.shape[:2]:
                mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        return self.apply(img_bgr, ckpt, cam=cam, mask=mask)

    def write_debug(self, frame_idx: int, out_path: Optional[Union[str, Path]] = None) -> Path:
        before, after, delta, lift = [], [], [], []

        for cam in sorted(self.frames_by_cam):
            img_path = self.frames_by_cam[cam][int(frame_idx)]
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(img_path)

            fixed = self.apply_file(img, cam, img_path=img_path)

            before.append(img)
            after.append(fixed)

            d = np.abs(fixed.astype(np.int16) - img.astype(np.int16)).mean(axis=2)
            delta.append(cv2.applyColorMap(np.clip(d * 4.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO))

            l0 = 0.114 * img[:, :, 0].astype(np.float32) + 0.587 * img[:, :, 1].astype(np.float32) + 0.299 * img[:, :, 2].astype(np.float32)
            l1 = 0.114 * fixed[:, :, 0].astype(np.float32) + 0.587 * fixed[:, :, 1].astype(np.float32) + 0.299 * fixed[:, :, 2].astype(np.float32)

            pos_lift = np.maximum(l1 - l0, 0.0)
            lift.append(cv2.applyColorMap(np.clip(pos_lift * 6.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO))

        strip = np.vstack([
            np.hstack(before),
            np.hstack(after),
            np.hstack(delta),
            np.hstack(lift),
        ])

        if out_path is None:
            out_path = self.recording_dir / f"color_match_debug_{int(frame_idx):06d}.jpg"

        out_path = Path(out_path)
        cv2.imwrite(str(out_path), strip)
        log(f"[write] {out_path}", "debug")
        return out_path


if __name__ == "__main__":
    cm = VideoColorMatcher(
        recording_dir="/home/charisoudis/capturestudio/data/Cagliari_2_5cams_Perf_1",
        hosts={
            0: {"cams": [1, 2, 4], "offset_ms": 0, "is_master": True},
            1: {"cams": [3, 5], "offset_ms": -1116},
        },
        sample_frames=30,
        fg_pixels=8000,
        bg_pixels=4000,
        max_pixels=200_000,
        steps=5000,
        batch=16384,
        lr=1e-3,
    )
    cm.fit_all()
    cm.write_debug(frame_idx=500)
