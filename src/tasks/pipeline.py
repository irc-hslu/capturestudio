from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from utils.misc import log


@dataclass(frozen=True)
class TaskSpec:
    """
    Single instantiation of a Celery task.
    - name: fully-qualified task name registered in Celery (e.g. "download.download_from_nas")
    - args/kwargs: call arguments for this signature
    - options: signature options (e.g., queue, countdown, expires, routing_key, etc.)
    """
    name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)  # e.g. {"queue": "cpu"}
    immutable: bool = True  # True -> .si, False -> .s


@dataclass(frozen=True)
class ChainSpec:
    parts: Sequence["SigSpec"]


@dataclass(frozen=True)
class GroupSpec:
    parts: Sequence["SigSpec"]


@dataclass(frozen=True)
class ChordSpec:
    header: Union[GroupSpec, Sequence["SigSpec"]]  # header must execute in parallel
    body: "SigSpec"


SigSpec = Union[TaskSpec, ChainSpec, GroupSpec, ChordSpec]


@dataclass(frozen=True)
class Stage:
    name: str
    parts: Sequence[SigSpec]


@dataclass
class Pipeline:
    stages: List[Stage]

    def to_celery(self, app):
        """Compile to chain(group(...), group(...), ...), preserving all nesting."""
        from celery import chain, group, chord

        def compile_sig(spec: SigSpec):
            if isinstance(spec, TaskSpec):
                if spec.name not in app.tasks:
                    raise KeyError(f"Task '{spec.name}' is not registered in the Celery app.")
                sig = app.tasks[spec.name].si(*spec.args, **spec.kwargs) if spec.immutable \
                    else app.tasks[spec.name].s(*spec.args, **spec.kwargs)
                return sig.set(**spec.options) if spec.options else sig

            if isinstance(spec, ChainSpec):
                if not spec.parts:
                    raise ValueError("ChainSpec cannot be empty.")
                return chain(*(compile_sig(p) for p in spec.parts))

            if isinstance(spec, GroupSpec):
                if not spec.parts:
                    raise ValueError("GroupSpec cannot be empty.")
                return group(*(compile_sig(p) for p in spec.parts))

            if isinstance(spec, ChordSpec):
                # Ensure header is a group signature
                hdr = spec.header if isinstance(spec.header, GroupSpec) else GroupSpec(spec.header)
                hdr_sig = compile_sig(hdr)  # -> group(...)
                body_sig = compile_sig(spec.body)  # -> task/chain/group
                return chord(hdr_sig, body_sig)

        if not self.stages:
            raise ValueError("Pipeline has no stages.")

        compiled_stage_groups = []
        for stage in self.stages:
            if not stage.parts:
                raise ValueError(f"Stage '{stage.name}' has no parts.")
            compiled = [compile_sig(p) for p in stage.parts]
            compiled_stage_groups.append(group(*compiled))

        return chain(*compiled_stage_groups)

    def to_svg(self, svg_path: Union[Path, str]) -> str:
        @dataclass
        class Theme:
            margin_x: int = 200
            margin_y: int = 80  # more top/bottom pad → vertical centering headroom
            lane_gap: int = 250
            stage_height: int = 420  # taller stages → longer vertical lines
            stage_gap: int = 80
            box_w: int = 200
            box_h: int = 42
            box_rx: int = 10
            rail_thick: int = 2
            line_thick: int = 2
            diag_len: int = 32
            barrier_stub: int = 100
            barrier_pad: int = 10
            font_size: int = 12
            stage_label_font_size: int = 15
            font_family: str = "JetBrains Mono"
            web_font_url: str | None = None  # set to a .woff2 URL to embed
            stage_label_dx: int = 40

            # per-item spacing inside chains/groups
            chain_item_min_h: int = 20  # min vertical space per chain element
            item_pad_top: int = 0  # top padding before a box
            item_pad_between: int = 0  # vertical gap between consecutive items

            color_text: str = "#0f172a"
            color_rail: str = "#475569"
            color_line: str = "#64748b"
            color_barrier: str = "#f59e0b"
            color_box_fill: str = "#ffffff"
            color_box_stroke: str = "#94a3b8"

            # Stage palette will be generated; these are fallbacks if needed
            default_stage_bg: str = "#f8fafc"
            default_stage_border: str = "#e2e8f0"

        def make_stage_palette(n: int) -> List[Tuple[str, str]]:
            """
            Return n (bg, border) pairs. Border is a darker variant of bg.
            Uses evenly spaced HSL hues with light tints.
            """
            out = []
            for i in range(max(1, n)):
                h = (i * 0.13) % 1.0  # rotate hue
                s = 0.35
                l_bg = 0.95
                l_border = 0.70
                r_bg, g_bg, b_bg = [int(c * 255) for c in colorsys.hls_to_rgb(h, l_bg, s)]
                r_bd, g_bd, b_bd = [int(c * 255) for c in colorsys.hls_to_rgb(h, l_border, s)]
                bg = f"#{r_bg:02x}{g_bg:02x}{b_bg:02x}"
                bd = f"#{r_bd:02x}{g_bd:02x}{b_bd:02x}"
                out.append((bg, bd))
            return out

        def lanes_needed(spec: SigSpec) -> int:
            if isinstance(spec, TaskSpec):
                return 1
            if isinstance(spec, ChainSpec):
                # width is the max width of any part in the chain
                return max(1, max(lanes_needed(p) for p in spec.parts) if spec.parts else 1)
            if isinstance(spec, GroupSpec):
                return sum(lanes_needed(p) for p in spec.parts)
            if isinstance(spec, ChordSpec):
                h = lanes_needed(spec.header if isinstance(spec.header, GroupSpec) else GroupSpec(spec.header))
                b = lanes_needed(spec.body)
                return max(h, b)
            raise TypeError(f"Unsupported spec: {type(spec)!r}")

        def lanes_for_stage(parts: Sequence[SigSpec]) -> int:
            # The stage behaves like a group of its parts (parallel execution per stage definition)
            return sum(lanes_needed(p) for p in parts)

        def last_token(task_name: str) -> str:
            return task_name.split(".")[-1] if "." in task_name else task_name

        def svg_escape(s: str) -> str:
            return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace('"', "&quot;").replace("'", "&apos;"))

        class SVG:
            def __init__(self, w: int, h: int, theme: Theme):
                self.w, self.h, self.t = w, h, theme
                self.parts: List[str] = []

            @staticmethod
            def font_to_data_uri(path: str, mime="font/woff"):
                b = Path(path).read_bytes()
                import base64
                return f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"

            def header(self):
                t = self.t
                # t.web_font_url = 'https://fonts.gstatic.com/s/jetbrainsmono/v23/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKxTOlOTk6OThhvA.woff'
                if t.web_font_url and False:
                    font_face = (
                            "@font-face{font-family:'PipelineCustom';src:url('%s') format('woff2');font-weight:400;font-style:normal;}"
                            % t.web_font_url
                    )
                    # Prepend the custom family to the stack
                    font_stack = f"PipelineCustom, {t.font_family}"
                else:
                    # font_stack = t.font_family
                    # data_uri = self.font_to_data_uri(str(PathUtils.font('JetBrainsMono-Regular.ttf')), mime="font/tff")
                    # font_face = (
                    #     "@font-face{font-family:'JetBrains Mono';"
                    #     f"src:url('{data_uri}') format('ttf');"
                    #     "font-weight:400;font-style:normal;}"
                    # )
                    font_stack = "'JetBrains Mono', monospace"

                self.parts.append(
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                    "<defs><style><![CDATA["
                    f'text{{font-family:{font_stack};font-size:{t.font_size}px;fill:{t.color_text};}}'
                    f'text{{font-family:{font_stack};font-size:{t.font_size}px;fill:{t.color_text};}}'
                    f'.box{{fill:{t.color_box_fill};stroke:{t.color_box_stroke};stroke-width:1;}}'
                    f'.rail{{stroke:{t.color_rail};stroke-width:{t.rail_thick};}}'
                    f'.line{{stroke:{t.color_line};stroke-width:{t.line_thick};fill:none;}}'
                    f'.barrier{{stroke:{t.color_barrier};stroke-width:{t.rail_thick};}}'
                    # Stage bg stroke color will vary per stage; we’ll set per-rect
                    "]]></style></defs>"
                )

            def add(self, s: str):
                self.parts.append(s)

            def finish(self) -> str:
                self.parts.append("</svg>")
                return "".join(self.parts)

        pipe = self
        theme: Theme = Theme()

        # --- measurements & canvas ---
        stage_lane_counts = [lanes_for_stage(s.parts) if s.parts else 1 for s in pipe.stages]
        max_lanes = max(stage_lane_counts or [1])

        w = theme.margin_x * 2 + max(1, max_lanes - 1) * theme.lane_gap + theme.box_w
        content_h = len(pipe.stages) * theme.stage_height + max(0, len(pipe.stages) - 1) * theme.stage_gap
        center_pad = int(theme.stage_height * 0.15)
        h = theme.margin_y * 2 + content_h + center_pad * 2

        svg = SVG(w, h, theme)
        svg.header()

        # global horizontal center (based on the widest stage)
        max_stage_w = (max(1, max_lanes) - 1) * theme.lane_gap + theme.box_w
        global_center_x = theme.margin_x + max_stage_w / 2

        # palette per stage
        palette = make_stage_palette(len(pipe.stages))

        # vertical placement start (center content)
        y = theme.margin_y + center_pad

        # previous stage connector info
        prev_center_x = None
        prev_bottom_y = None

        # spacing knobs
        chain_item_min_h = getattr(theme, "chain_item_min_h", 100)
        item_pad_top = getattr(theme, "item_pad_top", 16)
        item_pad_between = getattr(theme, "item_pad_between", 16)
        stage_label_font_size = getattr(theme, "stage_label_font_size", 16)
        barrier_pad = getattr(theme, "barrier_pad", 18)  # layout padding around barrier

        # --- helpers bound to this scope ---
        def svg_escape(s: str) -> str:
            return (s.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&apos;"))

        def last_token(task_name: str) -> str:
            return task_name.split(".")[-1] if "." in task_name else task_name

        def center_of_block(block_idxs: list[int], lane_x_stage: list[float]) -> float:
            xs = [lane_x_stage[i] for i in block_idxs]
            return sum(xs) / len(xs)

        def draw_task_box(cx: float, box_top_y: float, label: str) -> float:
            """
            Draw just the box; do NOT draw incoming/outgoing verticals.
            Return bottom y of the box.
            """
            bx = cx - theme.box_w / 2
            by = box_top_y
            svg.add(f'<rect class="box" x="{bx}" y="{by}" width="{theme.box_w}" height="{theme.box_h}" rx="{theme.box_rx}"/>')
            svg.add(f'<text x="{cx}" y="{by + theme.box_h / 2}" dominant-baseline="middle" text-anchor="middle">{svg_escape(label)}</text>')
            return by + theme.box_h

        def draw_barrier(ypos: float, x_left: float, x_right: float):
            svg.add(f'<line class="barrier" x1="{x_left}" y1="{ypos}" x2="{x_right}" y2="{ypos}"/>')

        def draw_split_trunk_to_branches(start_y: float, split_y: float, parent_x: float,
                                         child_xs: list[float], branch_start_y: float):
            """
            Vertical trunk (parent_x, start_y -> split_y), then diagonals to each child_x,
            then verticals down to branch_start_y (aligned start for all branches).
            """
            svg.add(f'<line class="line" x1="{parent_x}" y1="{start_y}" x2="{parent_x}" y2="{split_y}"/>')
            for cx in child_xs:
                svg.add(f'<line class="line" x1="{parent_x}" y1="{split_y}" x2="{cx}" y2="{split_y + theme.diag_len}"/>')
                # IMPORTANT: continue with a vertical spine to the branch start,
                # not into a box directly
                svg.add(f'<line class="line" x1="{cx}" y1="{split_y + theme.diag_len}" x2="{cx}" y2="{branch_start_y}"/>')

        # --- stage loop ---
        for sidx, stage in enumerate(pipe.stages):
            lanes = stage_lane_counts[sidx] or 1
            stage_w = (lanes - 1) * theme.lane_gap + theme.box_w
            sx = int(global_center_x - stage_w / 2)  # center this stage on global axis
            sy = y
            stage_h = theme.stage_height
            if 'preprocess' in stage.name:
                stage_h += 60
            else:
                stage_h -= 60
            ey = y + stage_h
            center_x = sx + stage_w / 2

            # per-stage lane x (centered under this stage)
            lane_x_stage = [sx + theme.box_w / 2 + i * theme.lane_gap for i in range(lanes)]

            # stage colors
            bg, border = palette[sidx] if sidx < len(palette) else (theme.default_stage_bg, theme.default_stage_border)

            # background + rails
            svg.add(
                f'<rect x="{sx - theme.box_w * 0.25}" y="{sy}" width="{stage_w + theme.box_w * 0.5}" '
                f'height="{stage_h}" rx="14" style="fill:{bg};stroke:{border};stroke-width:1"/>'
            )
            sy = sy + 10
            ey = ey - 10
            svg.add(f'<line class="rail" x1="{sx}" y1="{sy}" x2="{sx + stage_w}" y2="{sy}"/>')  # top rail
            svg.add(f'<line class="rail" x1="{sx}" y1="{ey}" x2="{sx + stage_w}" y2="{ey}"/>')  # bottom rail

            # vertical stage label, colored like border
            label_x = sx + stage_w + getattr(theme, "stage_label_dx", 24)
            label_y = sy + theme.stage_height / 2 + 20
            svg.add(
                f'<g transform="translate({label_x},{label_y}) rotate(-90)">'
                f'<text style="fill:{border};font-size:{stage_label_font_size}px">{svg_escape(stage.name)}</text>'
                f'</g>'
            )

            # connector from previous stage (center to center)
            if prev_center_x is not None and prev_bottom_y is not None:
                svg.add(f'<line class="line" x1="{prev_center_x}" y1="{prev_bottom_y}" x2="{center_x}" y2="{sy}"/>')

            # compute child blocks for this stage
            total_child = lanes_for_stage(stage.parts)
            if total_child <= 0:
                prev_center_x, prev_bottom_y = center_x, ey
                y = ey + theme.stage_gap
                continue

            blocks: list[list[int]] = []
            cursor = 0
            for part in stage.parts:
                n = lanes_needed(part)
                blocks.append(list(range(cursor, cursor + n)))
                cursor += n

            # branch x for each top-level part (center multi-lane blocks)
            child_xs = [center_of_block(b, lane_x_stage) for b in blocks]

            # aligned branch start inside the stage
            split_y = sy + 36
            branch_start_y = sy + 80

            # trunk → split → diagonals → vertical spines (no horizontals at stage start)
            draw_split_trunk_to_branches(sy, split_y, center_x, child_xs, branch_start_y)

            # --- recursive renderer (uses lane_x_stage) ---
            def render_spec(spec: SigSpec, lane_idxs: list[int], start_y: float, end_y: float, entry_from_x: float):
                height = end_y - start_y

                if isinstance(spec, TaskSpec):
                    cx = entry_from_x if entry_from_x is not None else lane_x_stage[lane_idxs[0]]
                    # 1) incoming vertical spine from start_y to box top padding
                    box_top = start_y + 10
                    svg.add(f'<line class="line" x1="{cx}" y1="{start_y}" x2="{cx}" y2="{box_top}"/>')
                    # 2) box
                    bottom = draw_task_box(cx, box_top, last_token(spec.name))
                    # 3) outgoing vertical spine to end_y
                    svg.add(f'<line class="line" x1="{cx}" y1="{bottom}" x2="{cx}" y2="{end_y}"/>')
                    return

                if isinstance(spec, ChainSpec):
                    cx = center_of_block(lane_idxs, lane_x_stage)
                    parts = spec.parts
                    if not parts:
                        svg.add(f'<line class="line" x1="{cx}" y1="{start_y}" x2="{cx}" y2="{end_y}"/>')
                        return
                    total_pad = item_pad_top + (len(parts) - 1) * item_pad_between
                    usable = max(0, height - total_pad)
                    per = max(chain_item_min_h, usable / len(parts))

                    ycursor = start_y + item_pad_top
                    for i, p in enumerate(parts):
                        sub_end = min(end_y, ycursor + per)
                        render_spec(p, lane_idxs[:], ycursor, sub_end, cx)
                        ycursor = sub_end + item_pad_between

                    if ycursor < end_y:
                        svg.add(f'<line class="line" x1="{cx}" y1="{ycursor}" x2="{cx}" y2="{end_y}"/>')
                    return

                if isinstance(spec, GroupSpec):
                    child_n = sum(lanes_needed(p) for p in spec.parts)
                    if child_n == 0:
                        cx = entry_from_x if entry_from_x is not None else center_of_block(lane_idxs, lane_x_stage)
                        svg.add(f'<line class="line" x1="{cx}" y1="{start_y}" x2="{cx}" y2="{end_y}"/>')
                        return

                    # contiguous blocks for each child
                    blocks_local: list[list[int]] = []
                    cur = 0
                    for p in spec.parts:
                        n = lanes_needed(p)
                        blocks_local.append(lane_idxs[cur:cur + n])
                        cur += n

                    parent_x = entry_from_x if entry_from_x is not None else center_of_block(lane_idxs, lane_x_stage)
                    xs_local = [center_of_block(b, lane_x_stage) for b in blocks_local]
                    split_y_local = start_y
                    branch_start_y_local = start_y + 40
                    draw_split_trunk_to_branches(start_y, split_y_local, parent_x, xs_local, branch_start_y_local)

                    for (p, b) in zip(spec.parts, blocks_local):
                        render_spec(p, b, branch_start_y_local, end_y, center_of_block(b, lane_x_stage))
                    return

                if isinstance(spec, ChordSpec):
                    header = spec.header if isinstance(spec.header, GroupSpec) else GroupSpec(spec.header)
                    header_w = sum(lanes_needed(p) for p in header.parts)
                    body_w = lanes_needed(spec.body)
                    width = max(header_w, body_w, 1)

                    idxs = lane_idxs[:width] if len(lane_idxs) != width else lane_idxs[:]
                    xs_all = [lane_x_stage[i] for i in idxs]
                    x_left, x_right = min(xs_all), max(xs_all)

                    # --- NEW: smarter barrier for group-of-chords → single-task body (reconstruction) ---
                    is_group_of_chords = isinstance(header, GroupSpec) and all(isinstance(p, ChordSpec) for p in header.parts)
                    is_single_task_body = isinstance(spec.body, TaskSpec)

                    height = end_y - start_y
                    if is_group_of_chords and is_single_task_body:
                        # Reserve enough room for the single body box + breathing space,
                        # then place the barrier right above that reserved space.
                        reserved_for_body = max(theme.box_h + 10, height * 0.28)
                        barrier_y = max(start_y + 40, end_y - reserved_for_body)
                    else:
                        # your existing behavior
                        barrier_y = start_y + height * 0.45 + 40  # + barrier_pad

                    # draw the barrier EXACTLY at barrier_y (do NOT add barrier_pad here)
                    draw_barrier(barrier_y, x_left - theme.barrier_stub / 2, x_right + theme.barrier_stub / 2)

                    parent_x = entry_from_x if entry_from_x is not None else center_of_block(idxs, lane_x_stage)

                    # HEADER (unchanged except it now targets header_end_y based on barrier_y)
                    if header_w > 0:
                        blocks_h, cur = [], 0
                        for hp in header.parts:
                            n = lanes_needed(hp);
                            blocks_h.append(idxs[cur:cur + n]);
                            cur += n
                        xs_h = [center_of_block(b, lane_x_stage) for b in blocks_h]

                        split_y_h = start_y
                        branch_start_y_h = start_y + 40
                        draw_split_trunk_to_branches(start_y, split_y_h, parent_x, xs_h, branch_start_y_h)

                        header_end_y = max(start_y, barrier_y - barrier_pad)
                        for (hp, hb, hx) in zip(header.parts, blocks_h, xs_h):
                            render_spec(hp, hb, branch_start_y_h, header_end_y, center_of_block(hb, lane_x_stage))
                            # ensure the branch vertical TOUCHES the barrier (no gap)
                            svg.add(f'<line class="line" x1="{hx}" y1="{header_end_y}" x2="{hx}" y2="{barrier_y}"/>')
                    else:
                        svg.add(f'<line class="line" x1="{parent_x}" y1="{start_y}" x2="{parent_x}" y2="{barrier_y}"/>')

                    # BODY → (use the centered-body block shown above)
                    body_start_y = barrier_y + barrier_pad
                    center_x_body = center_of_block(idxs, lane_x_stage)
                    if isinstance(spec.body, TaskSpec):
                        svg.add(f'<line class="line" x1="{center_x_body}" y1="{barrier_y}" x2="{center_x_body}" y2="{body_start_y}"/>')
                        render_spec(spec.body, idxs[:1], body_start_y, end_y, center_x_body)
                    else:
                        xs_body = [lane_x_stage[i] for i in idxs]
                        if isinstance(spec.body, GroupSpec):
                            bx = np.mean(xs_body)
                            svg.add(f'<line class="line" x1="{bx}" y1="{barrier_y}" x2="{bx}" y2="{body_start_y}"/>')
                        else:
                            for bx in xs_body:
                                svg.add(f'<line class="line" x1="{bx}" y1="{barrier_y}" x2="{bx}" y2="{body_start_y}"/>')
                        render_spec(spec.body, idxs, body_start_y, end_y, center_x_body)
                    return

                raise TypeError(f"Unsupported spec {type(spec)!r}")

            # render each top-level part from the aligned branch start
            for (part, block) in zip(stage.parts, blocks):
                render_spec(part, block, branch_start_y, ey, center_of_block(block, lane_x_stage))

            # remember for next stage connector
            prev_center_x, prev_bottom_y = center_x, ey
            y = ey + theme.stage_gap

        svg_text = svg.finish()
        svg_path = Path(svg_path)
        if svg_path.suffix in ['.pdf', '.png', '.eps']:
            import cairosvg
            getattr(cairosvg, f'svg2{svg_path.suffix[1:]}')(bytestring=svg_text.encode('utf-8'), write_to=str(svg_path))
        elif svg_path.suffix == '.svg':
            with open(svg_path.with_suffix('.svg'), "w", encoding="utf-8") as f:
                f.write(svg_text)
        else:
            raise ValueError(f"Unsupported SVG output format: {svg_path.suffix}. Use .svg or .pdf.")
        log(f'[{self.__class__.__name__}::to_svg] Pipeline plot exported to {svg_path}')
