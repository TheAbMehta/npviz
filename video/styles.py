
from __future__ import annotations

DARK_THEME = {
    "background": "#1a1a2e",
    "foreground": "#ecf0f1",
    "primary": "#2c3e50",
    "accent": "#e94560",
    "text": "#ecf0f1",
    "grid": "#2d2d44",
    "layer_fill": "#16213e",
    "layer_stroke": "#0f3460",
    "connection": "#bdc3c7",
    "skip_connection": "#3498db",
}

EVENT_ANIMATIONS = {
    "prune_head": {"color": "#e74c3c", "effect": "shrink_out"},
    "grow_head": {"color": "#2ecc71", "effect": "grow_in"},
    "prune_layer": {"color": "#c0392b", "effect": "collapse"},
    "grow_layer": {"color": "#27ae60", "effect": "expand"},
    "reconnect": {"color": "#3498db", "effect": "draw_line"},
}

QUALITY_PRESETS = {
    "low": {"width": 854, "height": 480, "frame_rate": 15},
    "medium": {"width": 1280, "height": 720, "frame_rate": 30},
    "high": {"width": 1920, "height": 1080, "frame_rate": 30},
    "4k": {"width": 3840, "height": 2160, "frame_rate": 60},
}


def importance_to_color(score: float, low: str = "#1a1a2e", high: str = "#ffdd40") -> str:
    score = max(0.0, min(1.0, score))

    def hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r1, g1, b1 = hex_to_rgb(low)
    r2, g2, b2 = hex_to_rgb(high)

    r = int(r1 + (r2 - r1) * score)
    g = int(g1 + (g2 - g1) * score)
    b = int(b1 + (b2 - b1) * score)

    return f"#{r:02x}{g:02x}{b:02x}"
