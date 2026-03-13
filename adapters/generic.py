
from __future__ import annotations

from .base import AdapterBase

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _resolve_attr(obj, dotted_path: str):
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def _find_largest_module_list(model) -> tuple[str, object] | None:
    if not _HAS_TORCH:
        return None
    best_name, best_mod, best_len = "", None, 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
            best_name, best_mod, best_len = name, mod, len(mod)
    if best_mod is not None:
        return best_name, best_mod
    return None


class GenericAdapter(AdapterBase):

    def __init__(self, model, layer_pattern: str | None = None):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for GenericAdapter")

        self.model = model
        self._layers = self._find_layers(model, layer_pattern)

    def _find_layers(self, model, pattern: str | None):
        if pattern is not None:
            layers = _resolve_attr(model, pattern)
            if not isinstance(layers, nn.ModuleList):
                layers = list(layers)
            return layers

        result = _find_largest_module_list(model)
        if result is None:
            raise ValueError(
                "Could not auto-detect layer ModuleList. "
                "Pass layer_pattern= explicitly."
            )
        return result[1]

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def heads_per_layer(self) -> list[int]:
        heads = []
        for layer in self._layers:
            n = self._detect_heads(layer)
            heads.append(n)
        return heads

    @property
    def params_per_layer(self) -> list[int]:
        return [sum(p.numel() for p in layer.parameters()) for layer in self._layers]

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _detect_heads(self, layer) -> int:
        for attr_path in [
            "self_attn.num_heads",
            "attn.num_heads",
            "attention.self.num_attention_heads",
            "self_attention.num_heads",
        ]:
            try:
                val = _resolve_attr(layer, attr_path)
                if isinstance(val, int):
                    return val
            except AttributeError:
                continue
        if hasattr(layer, "num_heads"):
            return layer.num_heads
        return 1

    def importance_scores(self) -> list[list[float]]:
        import torch

        scores = []
        for layer_idx, layer in enumerate(self._layers):
            n_heads = self.heads_per_layer[layer_idx]
            out_proj = None
            for attr_path in ["self_attn.out_proj", "attn.c_proj", "attention.output.dense"]:
                try:
                    out_proj = _resolve_attr(layer, attr_path)
                    break
                except AttributeError:
                    continue

            if out_proj is not None and hasattr(out_proj, "weight"):
                w = out_proj.weight.data
                d_model = w.shape[0]
                head_dim = d_model // n_heads if n_heads > 0 else d_model
                head_scores = []
                for h in range(n_heads):
                    start = h * head_dim
                    end = start + head_dim
                    if end <= w.shape[1]:
                        score = w[:, start:end].abs().mean().item()
                    else:
                        score = w.abs().mean().item()
                    if hasattr(out_proj, "weight_mask"):
                        mask_slice = out_proj.weight_mask[:, start:end]
                        if mask_slice.sum() == 0:
                            score = 0.0
                    head_scores.append(round(score, 6))
                scores.append(head_scores)
            else:
                scores.append([1.0] * n_heads)
        return scores
