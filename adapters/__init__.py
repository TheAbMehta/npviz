
from .base import AdapterBase

__all__ = ["AdapterBase", "auto_detect"]


def auto_detect(model) -> AdapterBase:
    try:
        from transformers import PreTrainedModel

        if isinstance(model, PreTrainedModel):
            from .transformers import TransformerAdapter

            return TransformerAdapter(model)
    except ImportError:
        pass

    try:
        from .generic import GenericAdapter

        return GenericAdapter(model)
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for model adapters. Install with: pip install npviz[adapters]"
        ) from e
