
from __future__ import annotations

from dash import Dash, Input, Output, State, callback_context

from ..schema import Timeline
from .. import plots


def register_callbacks(app: Dash, timeline: Timeline):
    max_step = max((s.step for s in timeline.snapshots), default=0)

    @app.callback(
        Output("step-slider", "max"),
        Output("step-slider", "value"),
        Output("step-slider", "marks"),
        Input("log-dir", "data"),
    )
    def init_slider(_):
        marks = {}
        for e in timeline.events:
            marks[e.step] = {"label": "", "style": {"color": plots.EVENT_COLORS.get(e.event_type, "gray")}}
        return max_step, 0, marks

    @app.callback(
        Output("timeline-graph", "figure"),
        Output("capacity-graph", "figure"),
        Output("stability-graph", "figure"),
        Input("log-dir", "data"),
    )
    def render_static_plots(_):
        return (
            plots.architecture_timeline(timeline),
            plots.capacity_allocation(timeline),
            plots.stability_overlay(timeline),
        )

    @app.callback(
        Output("heatmap-graph", "figure"),
        Output("topology-graph", "figure"),
        Input("step-slider", "value"),
    )
    def update_step_views(step):
        snap = timeline.snapshot_at(step)
        imp = timeline.importance_at(step)
        return (
            plots.importance_heatmap(imp, snap, step),
            plots.topology_view(snap, step),
        )

    @app.callback(
        Output("event-details", "children"),
        Input("timeline-graph", "clickData"),
    )
    def show_event_details(click_data):
        if not click_data or not click_data.get("points"):
            return "Click an event on the timeline to see details."

        point = click_data["points"][0]
        step = point.get("x")
        events_at_step = [e for e in timeline.events if e.step == step]

        if not events_at_step:
            return f"No event found at step {step}."

        lines = []
        for e in events_at_step:
            lines.append(f"Step: {e.step}")
            lines.append(f"Type: {e.event_type}")
            lines.append(f"Layer: {e.layer_idx}" + (f", Head: {e.head_idx}" if e.head_idx is not None else ""))
            lines.append(f"Reason: {e.reason}")
            if e.importance_score is not None:
                lines.append(f"Importance: {e.importance_score:.4f}")
            lines.append(f"Loss before: {e.loss_before:.4f}")
            if e.loss_after is not None:
                lines.append(f"Loss after:  {e.loss_after:.4f}")
                delta = e.loss_after - e.loss_before
                lines.append(f"Loss delta:  {delta:+.4f} ({'worse' if delta > 0 else 'better'})")
            lines.append("")

        return "\n".join(lines)

    @app.callback(
        Output("step-slider", "value", allow_duplicate=True),
        Input("timeline-graph", "clickData"),
        prevent_initial_call=True,
    )
    def snap_slider_to_event(click_data):
        if not click_data or not click_data.get("points"):
            from dash import no_update
            return no_update
        return click_data["points"][0].get("x", 0)
