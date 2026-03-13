
from __future__ import annotations

from .base import AdapterBase

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from transformers import PreTrainedModel

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

_LAYER_REGISTRY: dict[str, str] = {
    "bert": "bert.encoder.layer",
    "roberta": "roberta.encoder.layer",
    "gpt2": "transformer.h",
    "gpt_neo": "transformer.h",
    "gpt_neox": "gpt_neox.layers",
    "llama": "model.layers",
    "mistral": "model.layers",
    "t5": "encoder.block",
    "opt": "model.decoder.layers",
    "falcon": "transformer.h",
}

_ATTN_OUT_REGISTRY: dict[str, str] = {
    "bert": "attention.output.dense",
    "roberta": "attention.output.dense",
    "gpt2": "attn.c_proj",
    "llama": "self_attn.o_proj",
    "mistral": "self_attn.o_proj",
    "t5": "layer.0.SelfAttention.o",
    "opt": "self_attn.out_proj",
}


def _resolve_attr(obj, dotted_path: str):
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


class TransformerAdapter(AdapterBase):

    def __init__(self, model):
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers is required for TransformerAdapter")
        if not _HAS_TORCH:
            raise ImportError("torch is required for TransformerAdapter")
        if not isinstance(model, PreTrainedModel):
            raise TypeError(f"Expected PreTrainedModel, got {type(model).__name__}")

        self.model = model
        self.config = model.config
        self._model_type = getattr(self.config, "model_type", "")
        self._layers = self._find_layers()

    def _find_layers(self):
        path = _LAYER_REGISTRY.get(self._model_type)
        if path:
            try:
                return _resolve_attr(self.model, path)
            except AttributeError:
                pass

        import torch.nn as nn

        best, best_len = None, 0
        for _, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
                best, best_len = mod, len(mod)
        if best is None:
            raise ValueError(f"Cannot find layers for model_type={self._model_type}")
        return best

    @property
    def n_layers(self) -> int:
        return getattr(self.config, "num_hidden_layers", len(self._layers))

    @property
    def heads_per_layer(self) -> list[int]:
        base_heads = getattr(self.config, "num_attention_heads", 1)
        pruned = getattr(self.config, "pruned_heads", {})

        heads = []
        for i in range(self.n_layers):
            n_pruned = len(pruned.get(i, pruned.get(str(i), [])))
            heads.append(base_heads - n_pruned)
        return heads

    @property
    def params_per_layer(self) -> list[int]:
        return [sum(p.numel() for p in layer.parameters()) for layer in self._layers]

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def importance_scores(self) -> list[list[float]]:
        scores = []
        attn_path = _ATTN_OUT_REGISTRY.get(self._model_type)

        for layer_idx, layer in enumerate(self._layers):
            n_heads = self.heads_per_layer[layer_idx]
            base_heads = getattr(self.config, "num_attention_heads", 1)

            out_proj = None
            if attn_path:
                try:
                    out_proj = _resolve_attr(layer, attn_path)
                except AttributeError:
                    pass

            if out_proj is not None and hasattr(out_proj, "weight"):
                w = out_proj.weight.data
                d_model = w.shape[0]
                head_dim = d_model // base_heads if base_heads > 0 else d_model

                pruned = getattr(self.config, "pruned_heads", {})
                pruned_set = set(pruned.get(layer_idx, pruned.get(str(layer_idx), [])))

                head_scores = []
                for h in range(base_heads):
                    if h in pruned_set:
                        continue
                    start = h * head_dim
                    end = start + head_dim

                    if hasattr(out_proj.weight, "grad") and out_proj.weight.grad is not None:
                        score = (w[:, start:end] * out_proj.weight.grad[:, start:end]).abs().mean().item()
                    else:
                        score = w[:, start:end].abs().mean().item()
                    head_scores.append(round(score, 6))
                scores.append(head_scores)
            else:
                scores.append([1.0] * n_heads)
        return scores
