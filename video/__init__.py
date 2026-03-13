
from __future__ import annotations

from pathlib import Path
from typing import Literal

SceneName = Literal["architecture", "importance", "overview"]
Quality = Literal["low", "medium", "high", "4k"]


def render_video(
    log_dir: str | Path,
    output: str | Path = "evolution.mp4",
    scene: SceneName = "overview",
    quality: Quality = "medium",
    theme: dict | None = None,
) -> Path:
    try:
        from manim import config as manim_config
        from manim import tempconfig
    except ImportError:
        raise ImportError(
            "manim is required for video generation. Install with: pip install npviz[video]"
        )

    from ..recorder import Recorder
    from .scenes import SCENE_REGISTRY
    from .styles import DARK_THEME, QUALITY_PRESETS

    output = Path(output)
    timeline = Recorder.load(log_dir)
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["medium"])
    theme = theme or DARK_THEME

    scene_cls = SCENE_REGISTRY.get(scene)
    if scene_cls is None:
        raise ValueError(f"Unknown scene: {scene!r}. Choose from: {list(SCENE_REGISTRY)}")

    with tempconfig({
        "pixel_width": preset["width"],
        "pixel_height": preset["height"],
        "frame_rate": preset["frame_rate"],
        "output_file": output.stem,
        "media_dir": str(output.parent / ".manim_media"),
    }):
        scene_instance = scene_cls(timeline, theme=theme)
        scene_instance.render()

        rendered = Path(manim_config.output_file).with_suffix(".mp4")
        if rendered.exists() and rendered != output:
            import shutil
            shutil.move(str(rendered), str(output))

    return output
