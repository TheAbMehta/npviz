
from dash import dcc, html


def timeline_section():
    return html.Div([
        html.H3("Architecture Timeline"),
        dcc.Graph(id="timeline-graph", config={"displayModeBar": False}),
        html.Div([
            html.Label("Training Step:"),
            dcc.Slider(id="step-slider", min=0, max=1, step=1, value=0,
                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "0 20px"}),
    ])


def heatmap_section():
    return html.Div([
        html.H3("Head Importance"),
        dcc.Graph(id="heatmap-graph"),
    ])


def topology_section():
    return html.Div([
        html.H3("Network Topology"),
        dcc.Graph(id="topology-graph"),
    ])


def capacity_section():
    return html.Div([
        html.H3("Capacity Allocation"),
        dcc.Graph(id="capacity-graph"),
    ])


def stability_section():
    return html.Div([
        html.H3("Training Stability"),
        dcc.Graph(id="stability-graph"),
    ])


def event_detail_section():
    return html.Div([
        html.H3("Event Details"),
        html.Div(id="event-details", children="Click an event on the timeline to see details.",
                 style={"padding": "10px", "backgroundColor": "#f8f9fa",
                        "borderRadius": "5px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
    ])
