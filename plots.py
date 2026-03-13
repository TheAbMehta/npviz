
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .schema import ArchSnapshot, ImportanceSnapshot, RewireEvent, Timeline


EVENT_COLORS = {
    "prune_head": "#e74c3c",
    "grow_head": "#2ecc71",
    "prune_layer": "#c0392b",
    "grow_layer": "#27ae60",
    "reconnect": "#3498db",
}

EVENT_SYMBOLS = {
    "prune_head": "triangle-down",
    "grow_head": "triangle-up",
    "prune_layer": "x",
    "grow_layer": "star",
    "reconnect": "diamond",
}


def architecture_timeline(timeline: Timeline) -> go.Figure:
    fig = go.Figure()

    if not timeline.events:
        fig.add_annotation(text="No rewiring events recorded", showarrow=False)
        return fig

    for etype in EVENT_COLORS:
        evts = [e for e in timeline.events if e.event_type == etype]
        if not evts:
            continue
        fig.add_trace(go.Scatter(
            x=[e.step for e in evts],
            y=[0] * len(evts),
            mode="markers",
            marker=dict(
                size=12,
                color=EVENT_COLORS[etype],
                symbol=EVENT_SYMBOLS.get(etype, "circle"),
            ),
            name=etype.replace("_", " ").title(),
            text=[f"L{e.layer_idx}" + (f"/H{e.head_idx}" if e.head_idx is not None else "")
                  + f"<br>{e.reason}" for e in evts],
            hovertemplate="Step %{x}<br>%{text}<extra></extra>",
            customdata=[i for i, e in enumerate(timeline.events) if e.event_type == etype],
        ))

    fig.update_layout(
        title="Architecture Timeline",
        xaxis_title="Training Step",
        yaxis=dict(visible=False, range=[-1, 1]),
        height=200,
        margin=dict(t=40, b=30, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
    )
    return fig


def importance_heatmap(imp: ImportanceSnapshot | None, snap: ArchSnapshot | None, step: int) -> go.Figure:
    fig = go.Figure()

    if imp is None or not imp.scores:
        fig.add_annotation(text=f"No importance data at step {step}", showarrow=False)
        fig.update_layout(height=350, margin=dict(t=40, b=30))
        return fig

    scores = imp.scores
    n_layers = len(scores)
    max_heads = max(len(row) for row in scores)

    import math
    padded = []
    for row in scores:
        padded.append(row + [math.nan] * (max_heads - len(row)))

    fig.add_trace(go.Heatmap(
        z=padded,
        colorscale=[
            [0.0, "#1a1a2e"],
            [0.3, "#16213e"],
            [0.5, "#0f3460"],
            [0.7, "#e94560"],
            [1.0, "#ffdd40"],
        ],
        colorbar=dict(title="Score"),
        hoverongaps=False,
        hovertemplate="Layer %{y}, Head %{x}<br>Importance: %{z:.4f}<extra></extra>",
    ))

    if snap:
        for layer_idx, n_heads in enumerate(snap.heads_per_layer):
            if layer_idx < n_layers:
                for h in range(n_heads, max_heads):
                    fig.add_shape(
                        type="rect",
                        x0=h - 0.5, x1=h + 0.5,
                        y0=layer_idx - 0.5, y1=layer_idx + 0.5,
                        fillcolor="black",
                        line=dict(width=0),
                    )

    fig.update_layout(
        title=f"Head Importance - Step {step}",
        xaxis_title="Head",
        yaxis_title="Layer",
        height=350,
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig


def topology_view(snap: ArchSnapshot | None, step: int) -> go.Figure:
    fig = go.Figure()

    if snap is None:
        fig.add_annotation(text=f"No snapshot at step {step}", showarrow=False)
        fig.update_layout(height=400, margin=dict(t=40, b=30))
        return fig

    max_heads = max(snap.heads_per_layer) if snap.heads_per_layer else 1

    for src, dst in snap.connections:
        if src < snap.n_layers and dst < snap.n_layers:
            fig.add_shape(
                type="line",
                x0=max_heads / 2, y0=src,
                x1=max_heads / 2, y1=dst,
                line=dict(color="#3498db", width=2, dash="dot"),
            )

    for i in range(snap.n_layers - 1):
        fig.add_shape(
            type="line",
            x0=max_heads / 2, y0=i,
            x1=max_heads / 2, y1=i + 1,
            line=dict(color="#bdc3c7", width=1),
        )

    for layer_idx, n_heads in enumerate(snap.heads_per_layer):
        offset = (max_heads - n_heads) / 2
        fig.add_trace(go.Scatter(
            x=[offset + h for h in range(n_heads)],
            y=[layer_idx] * n_heads,
            mode="markers",
            marker=dict(size=20, color="#2c3e50", line=dict(width=2, color="#ecf0f1")),
            name=f"Layer {layer_idx}",
            showlegend=False,
            hovertemplate=f"Layer {layer_idx}, Head %{{x:.0f}}<br>Params: {snap.params_per_layer[layer_idx]:,}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Topology - Step {step} ({snap.n_layers} layers, {snap.total_params:,} params)",
        xaxis=dict(visible=False, range=[-1, max_heads]),
        yaxis=dict(title="Layer", dtick=1, range=[-0.5, snap.n_layers - 0.5]),
        height=400,
        margin=dict(t=40, b=30, l=40, r=20),
        plot_bgcolor="white",
    )
    return fig


def capacity_allocation(timeline: Timeline) -> go.Figure:
    fig = go.Figure()

    if not timeline.snapshots:
        fig.add_annotation(text="No snapshots recorded", showarrow=False)
        fig.update_layout(height=300, margin=dict(t=40, b=30))
        return fig

    snaps = sorted(timeline.snapshots, key=lambda s: s.step)
    steps = [s.step for s in snaps]
    n_layers = max(s.n_layers for s in snaps)

    for layer_idx in range(n_layers):
        params = []
        for s in snaps:
            if layer_idx < len(s.params_per_layer):
                params.append(s.params_per_layer[layer_idx])
            else:
                params.append(0)
        fig.add_trace(go.Scatter(
            x=steps,
            y=params,
            mode="lines",
            stackgroup="one",
            name=f"Layer {layer_idx}",
            hovertemplate=f"Layer {layer_idx}<br>Step %{{x}}<br>Params: %{{y:,}}<extra></extra>",
        ))

    fig.update_layout(
        title="Capacity Allocation Over Time",
        xaxis_title="Training Step",
        yaxis_title="Parameters",
        height=300,
        margin=dict(t=40, b=30, l=60, r=20),
    )
    return fig


def stability_overlay(timeline: Timeline) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    if timeline.metrics:
        metrics = sorted(timeline.metrics, key=lambda m: m.step)
        steps = [m.step for m in metrics]
        fig.add_trace(go.Scatter(
            x=steps,
            y=[m.loss for m in metrics],
            mode="lines",
            name="Loss",
            line=dict(color="#2c3e50"),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=steps,
            y=[m.grad_norm for m in metrics],
            mode="lines",
            name="Grad Norm",
            line=dict(color="#8e44ad"),
        ), row=2, col=1)

    for event in timeline.events:
        color = EVENT_COLORS.get(event.event_type, "gray")
        for row in [1, 2]:
            fig.add_vline(
                x=event.step, line=dict(color=color, width=1, dash="dash"),
                row=row, col=1,
            )

    fig.update_layout(
        title="Training Stability",
        height=350,
        margin=dict(t=40, b=30, l=60, r=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Grad Norm", row=2, col=1)
    fig.update_xaxes(title_text="Training Step", row=2, col=1)
    return fig
