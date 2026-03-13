
from .recorder import Recorder
from .schema import Timeline
from .viewer import serve


def __getattr__(name):
    if name == "adapters":
        from . import adapters
        return adapters
    if name == "auto_detect":
        from .adapters import auto_detect
        return auto_detect
    if name == "export_run":
        from .export import export_run
        return export_run
    if name == "render_video":
        from .video import render_video
        return render_video
    raise AttributeError(f"module 'npviz' has no attribute {name!r}")


__all__ = ["Recorder", "Timeline", "serve", "adapters", "auto_detect", "export_run", "render_video"]
