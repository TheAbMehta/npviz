
from __future__ import annotations

from pathlib import Path

from dash import Dash, html, dcc

from ..recorder import Recorder
from ..schema import Timeline
from . import components
from .callbacks import register_callbacks


def create_app(log_dir: str | Path) -> Dash:
    timeline = Recorder.load(log_dir)

    app = Dash(__name__, title="npviz - Neuroplastic Architecture Viewer")
    app.layout = html.Div([
        dcc.Store(id="log-dir", data=str(log_dir)),

        html.H1("Neuroplastic Architecture Visualizer",
                style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "5px"}),
        html.P(f"Run: {Path(log_dir).name} - {len(timeline.events)} rewiring events, "
               f"{len(timeline.snapshots)} snapshots",
               style={"textAlign": "center", "color": "#7f8c8d", "marginTop": "0"}),

        html.Hr(),

        html.Div([
            html.Div([components.timeline_section()], style={"flex": "2"}),
            html.Div([components.event_detail_section()], style={"flex": "1", "padding": "0 20px"}),
        ], style={"display": "flex", "marginBottom": "20px"}),

        html.Div([
            html.Div([components.heatmap_section()], style={"flex": "1"}),
            html.Div([components.topology_section()], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

        html.Div([
            html.Div([components.capacity_section()], style={"flex": "1"}),
            html.Div([components.stability_section()], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px"}),

    ], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px",
              "fontFamily": "system-ui, -apple-system, sans-serif"})

    register_callbacks(app, timeline)
    return app


def serve(log_dir: str | Path, port: int = 8050, debug: bool = False):
    app = create_app(log_dir)
    app.run(host="0.0.0.0", port=port, debug=debug)
