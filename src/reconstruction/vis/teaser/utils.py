import os
from pathlib import Path

os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'

import numpy as np
from moviepy.editor import (
    VideoFileClip,
    VideoClip,
    vfx,
    concatenate_videoclips,
    clips_array
)

from utils.misc import PathUtils, log


def create_teaser_rgb_depth_normals_video(cfg, force=False):
    if Path(cfg["output"]).exists() and not force:
        log(f"Teaser video {cfg['output']} already exists. Skipping creation.", 'debug')
        return

    def grid(h, w):
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        s = (w - 1 - x) + y                    # 0 @ top-right, L @ bottom-left
        return s, (w + h - 2)


    def ease(t, d):
        return 0.5 * (1 - np.cos(np.pi * np.clip(t / d, 0, 1)))


    def get_frame_safe(clip, t):
        """Clamp t so we never wrap around."""
        return clip.get_frame(min(t, clip.duration - 1.0 / clip.fps))

    # ── load / normalise ───────────────────────────────────────────── #
    fps = cfg["fps"]
    h_target = cfg["target_height"]

    rgb   = VideoFileClip(cfg["paths"]["rgb"  ]).fx(vfx.resize, height=h_target).set_fps(fps)
    depth = VideoFileClip(cfg["paths"]["depth"]).fx(vfx.resize, rgb.size      ).set_fps(fps)
    norm  = VideoFileClip(cfg["paths"]["norm" ]).fx(vfx.resize, rgb.size      ).set_fps(fps)

    # timeline lengths
    dur = cfg["durations"]
    dA, dB, dDH, dC, dD, dE, dF = dur.values()
    total_len = sum(dur.values())

    # trim all sources to cover the full teaser
    rgb, depth, norm = (c.subclip(0, total_len) for c in (rgb, depth, norm))

    h, w = rgb.h, rgb.w
    s, L = grid(h, w)

    # boundaries from band proportions
    p_rgb = cfg["bands"]["rgb_corner"]
    p_n   = cfg["bands"]["normal"]
    p_d   = cfg["bands"]["depth"]

    B1, B2, B3 = p_rgb * L, (p_rgb + p_n) * L, (p_rgb + p_n + p_d) * L

    phases = []
    t_start = 0.0  # absolute start of next phase

    # helper to freeze t_start per phase
    def mkclip(dur, fn):
        t0 = t_start

        def wrapped(t):
            return fn(t, t + t0).astype("uint8")

        return VideoClip(wrapped, duration=dur).set_fps(fps)

    # ── Phase A : RGB only ─────────────────────────────────────────── #
    phases.append(rgb.subclip(0, dA))
    t_start += dA

    # ── Phase B : RGB → Depth ──────────────────────────────────────── #
    def fB(t, tg):
        m = (s <= ease(t, dB) * L)[..., None]
        return (1 - m) * get_frame_safe(rgb, tg) + m * get_frame_safe(depth, tg)

    phases.append(mkclip(dB, fB))
    t_start += dB

    # ── DEPTH-ONLY HOLD ───────────────────────────────────────────── #
    dDH = cfg["durations"]["depth_hold"]

    def fDH(_, tg):
        # depth is full-screen for this pause
        return get_frame_safe(depth, tg)

    phases.append(mkclip(dDH, fDH))
    t_start += dDH

    # ── Phase C : Depth → Normal ───────────────────────────────────── #
    def fC(t, tg):
        m = (s <= ease(t, dC) * L)[..., None]
        return (1 - m) * get_frame_safe(depth, tg) + m * get_frame_safe(norm, tg)

    phases.append(mkclip(dC, fC))
    t_start += dC

    # ── Phase D : simultaneous double-corner reveal ────────────────── #
    def fD(t, tg):
        p = ease(t, dD)
        b_top = p * B1
        b_dep = L - p * (L - B2)
        b_bot = L - p * (L - B3)

        m_rgb_tr = s < b_top
        m_rgb_bl = s >= b_bot
        m_depth  = (s >= b_dep) & (s < b_bot)
        m_norm   = ~(m_rgb_tr | m_rgb_bl | m_depth)

        return (
            m_rgb_tr[..., None] * get_frame_safe(rgb, tg)
            + m_rgb_bl[..., None] * get_frame_safe(rgb, tg)
            + m_depth [..., None] * get_frame_safe(depth, tg)
            + m_norm  [..., None] * get_frame_safe(norm, tg)
        )

    phases.append(mkclip(dD, fD))
    t_start += dD

    # ── Phase E : hold tableau ─────────────────────────────────────── #
    def fE(_, tg):
        m_rgb_tr = s < B1
        m_rgb_bl = s >= B3
        m_depth  = (s >= B2) & (s < B3)
        m_norm   = (s >= B1) & (s < B2)

        return (
            (m_rgb_tr | m_rgb_bl)[..., None] * get_frame_safe(rgb, tg)
            + m_depth[..., None] * get_frame_safe(depth, tg)
            + m_norm [..., None] * get_frame_safe(norm, tg)
        )

    phases.append(mkclip(dE, fE))
    t_start += dE

    # ── Phase F : collapse bands back to TR ────────────────────────── #
    def fF(t, tg):
        q = ease(t, dF)
        b1 = (1 - q) * B1
        b2 = b1 + (1 - q) * p_n * L
        b3 = b2 + (1 - q) * p_d * L

        m_norm  = (s >= b1) & (s < b2)
        m_depth = (s >= b2) & (s < b3)
        m_rgb   = ~(m_norm | m_depth)

        return (
            m_rgb  [..., None] * get_frame_safe(rgb, tg)
            + m_depth[..., None] * get_frame_safe(depth, tg)
            + m_norm [..., None] * get_frame_safe(norm, tg)
        )

    phases.append(mkclip(dF, fF))

    # ── render ─────────────────────────────────────────────────────── #
    concatenate_videoclips(phases).write_videofile(
        cfg["output"],
        codec="libx264",
        bitrate=cfg["bitrate"],
        audio=False,
        fps=fps,
        threads=4,
    )

def create_teaser_grid(cfg, force=False):
    if Path(cfg["output"]).exists() and not force:
        log(f"Teaser grid video {cfg['output']} already exists. Skipping creation.", 'debug')
        return

    def label_clip(base_clip, txt, cfg):
        """
        Overlay `txt` in the top-left corner using Pillow instead of ImageMagick.
        Falls back to a default font if the requested one isn’t found.
        """
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        from moviepy.editor import ImageClip, CompositeVideoClip

        # 1. Pick a font (or default)
        try:
            font = ImageFont.truetype(str(PathUtils.font('JetBrainsMono-Regular.ttf')), cfg["font_size"])
        except IOError:
            font = ImageFont.load_default(cfg["font_size"])

        # 2. Render the text on a small RGBA canvas
        pad = 6
        bbox = font.getbbox(txt)
        w_txt, h_txt = bbox[2] - bbox[0], bbox[3] - bbox[1]+4
        canvas = Image.new("RGBA", (int(w_txt + 2 * pad), int(h_txt + 2 * pad)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # Outline for contrast
        sc, sw = cfg["stroke_color"], cfg["stroke_width"]
        if sw > 0:
            for dx in range(-sw, sw + 1):
                for dy in range(-sw, sw + 1):
                    draw.text((pad + dx, pad + dy), txt, font=font, fill=sc)

        draw.text((pad, pad), txt, font=font, fill=cfg["font_color"])

        # 3. Convert to an ImageClip with the same duration as the base clip
        txt_clip = (
            ImageClip(np.array(canvas))
            .set_duration(base_clip.duration)
            .set_position(("left", "top"))
        )

        # 4. Composite and return
        return CompositeVideoClip([base_clip, txt_clip])

    # W_out, H_out = cfg["out_res"]
    fps = cfg["fps"]

    # ── 1. Load all six clips ──────────────────────────────────────────
    clips = [VideoFileClip(p).set_fps(fps) for p in cfg["paths"]]

    # All clips should play the same length in a grid.
    # You can choose min(...) to clip the longer ones, or max(...) and .loop() the shorter ones.
    grid_duration = min(c.duration for c in clips)
    clips = [c.subclip(0, grid_duration) for c in clips]

    # # ── 2. Resize each to its cell size ────────────────────────────────
    # W_cell = W_out // 2
    # H_cell = H_out // 2
    # clips = [c.fx(vfx.resize, (W_cell, H_cell)) for c in clips]

    # ── 3. Burn-in labels ──────────────────────────────────────────────
    clips = [
        label_clip(c, lbl, cfg) for c, lbl in zip(clips, cfg["labels"])
    ]

    # Arrange as 2 rows × 3 columns (row-major)
    if len(clips) == 8:
        grid = clips_array([[clips[0], clips[1], clips[2], clips[3]],
                            [clips[4], clips[5], clips[6], clips[7]]])
    elif len(clips) == 6:
        grid = clips_array([[clips[0], clips[1], clips[2]],
                            [clips[3], clips[4], clips[5]]])
    elif len(clips) <=5:
        grid = clips_array([clips])

    # ── 4. Export ─────────────────────────────────────────────────────
    grid.write_videofile(
        cfg["output"],
        codec="libx264",
        bitrate=cfg["bitrate"],
        fps=fps,
        audio=False,
    )


if __name__ == "__main__":
    video_paths_, labels_ = [], []
    for MODALITY_ in ['pcd', 'gs']:
        for DEPTH_ in ['bilateral_spatial', 'bilateral_temporal', 'stereo|raftstereo', 'stereo|foundationstereo']:
            if DEPTH_.startswith('stereo'):
                _, PREFIX_ = DEPTH_.split('|')
                MODALITY_ = f'stereo{"+gs" if MODALITY_ == "gs" else ""}_split'
            else:
                PREFIX_ = DEPTH_
            out_path_ = PathUtils().out_path() / 'results' / 'pcd' /  f"teaser_{MODALITY_}_{PREFIX_}.mp4"
            cfg_ = {
                # source files
                "paths": {
                    "rgb": str(PathUtils().out_path() / 'results' / 'pcd' / f"{PREFIX_}_rgb_rgb_{MODALITY_}.mp4"),
                    "depth": str(PathUtils().out_path() / 'results' / 'pcd' / f"{PREFIX_}_depth_depth_{MODALITY_}.mp4"),
                    "norm": str(PathUtils().out_path() / 'results' / 'pcd' / f"{PREFIX_}_feat_feat_{MODALITY_}.mp4"),
                },

                # output & global video params
                "output": str(out_path_),
                "fps": 30,
                "target_height": 720,
                "bitrate": "15M",

                # timeline (seconds) ─ names are free; order matters
                "durations": {
                    "rgb_only": 1.5,  # A
                    "rgb_to_depth": 1.0,  # B  (transition)
                    "depth_hold": 1.0,  # NEW  ← depth full-screen pause
                    "depth_to_norm": 1.5,  # C  (transition)
                    "reveal_bands": 1.5,  # D
                    "hold": 1.5,  # E
                    "collapse": 1.0,  # F
                },

                # diagonal proportions (must sum ≤ 1.0; remaining goes to RGB corners)
                "bands": {
                    "rgb_corner": 0.35,  # each corner
                    "normal": 0.15,
                    "depth": 0.15,
                },
            }
            create_teaser_rgb_depth_normals_video(cfg_)
            if out_path_.exists():
                video_paths_.append(out_path_)
                labels_.append(f'{MODALITY_} {PREFIX_}'.replace("_split","").replace("_", " ").upper())

    GRID_CFG = {
        # Six input videos, row-major order: [row0-col0, row0-col1, …]
        "paths": [str(p) for p in video_paths_],

        # Short labels that will be burned into the clips (same order)
        "labels": labels_,

        # Final grid resolution (width, height). 1920×1080 → each cell ≈ 640×540.
        "out_res": (3072, 2560),

        # Font & styling for overlay text
        "font": "JetBrainsMono-Regular.ttf",
        "font_size": 32,
        "font_color": "white",
        "stroke_color": "black",
        "stroke_width": 0,

        # Output file & encoding
        "output": str(PathUtils().out_path() / 'results' / 'pcd' /  f"teaser_grid.mp4"),
        "fps": 30,
        "bitrate": "15M",
    }
    create_teaser_grid(GRID_CFG)
