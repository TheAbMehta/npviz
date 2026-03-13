
from __future__ import annotations

from pathlib import Path
from typing import Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import (
    architecture_timeline,
    capacity_allocation,
    importance_heatmap,
    stability_overlay,
    topology_view,
)
from .recorder import Recorder
from .schema import Timeline

SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.0
DEFAULT_DPI = 300
DEFAULT_FONT_FAMILY = "Computer Modern, serif"
DEFAULT_FONT_SIZE = 10
TITLE_FONT_SIZE = 12

Format = Literal["svg", "pdf", "png"]


def apply_paper_style(
    fig: go.Figure,
    font_family: str = DEFAULT_FONT_FAMILY,
    font_size: int = DEFAULT_FONT_SIZE,
) -> go.Figure:
    fig.update_layout(
        font=dict(family=font_family, size=font_size),
        title_font_size=TITLE_FONT_SIZE,
        plot_bgcolor="white",
        paper_bgcolor="white",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, ticks="outside",
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="black",
        mirror=True, ticks="outside",
    )
    return fig


def export_figure(
    fig: go.Figure,
    path: str | Path,
    fmt: Format = "svg",
    width: int | None = None,
    height: int | None = None,
    scale: int = 2,
) -> Path:
    try:
        import kaleido
    except ImportError:
        raise ImportError("kaleido is required for figure export. Install with: pip install npviz[export]")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {"format": fmt, "scale": scale}
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height

    fig.write_image(str(path), **kwargs)
    return path


def export_panel(
    timeline: Timeline,
    output_dir: str | Path,
    fmt: Format = "svg",
    step: int | None = None,
    paper_style: bool = True,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if step is None:
        step = timeline.steps[-1] if timeline.steps else 0

    snap = timeline.snapshot_at(step)
    imp = timeline.importance_at(step)

    figures = {
        "timeline": architecture_timeline(timeline),
        "heatmap": importance_heatmap(imp, snap, step),
        "topology": topology_view(snap, step),
        "capacity": capacity_allocation(timeline),
        "stability": stability_overlay(timeline),
    }

    if paper_style:
        for fig in figures.values():
            apply_paper_style(fig)

    paths = []
    width_px = int(DOUBLE_COL_WIDTH * DEFAULT_DPI)
    for name, fig in figures.items():
        p = export_figure(fig, output_dir / f"{name}.{fmt}", fmt=fmt, width=width_px)
        paths.append(p)

    panel = _build_panel(timeline, snap, imp, step, paper_style)
    panel_path = export_figure(
        panel,
        output_dir / f"panel.{fmt}",
        fmt=fmt,
        width=int(DOUBLE_COL_WIDTH * DEFAULT_DPI),
        height=int(DOUBLE_COL_WIDTH * 1.2 * DEFAULT_DPI),
    )
    paths.append(panel_path)

    return paths


def _build_panel(
    timeline: Timeline, snap, imp, step: int, paper_style: bool,
) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Architecture Timeline", "Head Importance",
            "Topology", "Capacity Allocation",
            "Training Stability", "",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
    )

    tl_fig = architecture_timeline(timeline)
    for trace in tl_fig.data:
        fig.add_trace(trace, row=1, col=1)

    hm_fig = importance_heatmap(imp, snap, step)
    for trace in hm_fig.data:
        fig.add_trace(trace, row=1, col=2)

    tp_fig = topology_view(snap, step)
    for trace in tp_fig.data:
        fig.add_trace(trace, row=2, col=1)

    ca_fig = capacity_allocation(timeline)
    for trace in ca_fig.data:
        fig.add_trace(trace, row=2, col=2)

    st_fig = stability_overlay(timeline)
    for trace in st_fig.data:
        fig.add_trace(trace, row=3, col=1)

    fig.update_layout(
        height=1200,
        showlegend=False,
        margin=dict(t=40, b=30, l=40, r=20),
    )

    if paper_style:
        apply_paper_style(fig)

    return fig


def export_run(
    log_dir: str | Path,
    output_dir: str | Path,
    fmt: Format = "svg",
    step: int | None = None,
    paper_style: bool = True,
) -> list[Path]:
    timeline = Recorder.load(log_dir)
    return export_panel(timeline, output_dir, fmt=fmt, step=step, paper_style=paper_style)
