
from __future__ import annotations

from .base import AdapterBase

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import torch_pruning as tp

    _HAS_TP = True
except ImportError:
    _HAS_TP = False


class TorchPruningAdapter(AdapterBase):

    def __init__(self, model, layer_pattern: str | None = None, pruner=None, importance=None):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for TorchPruningAdapter")

        self.model = model
        self.pruner = pruner
        self.importance_fn = importance
        self._layers = self._find_layers(model, layer_pattern)
        self._cached_heads: list[int] | None = None

    def _find_layers(self, model, pattern: str | None):
        if pattern is not None:
            obj = model
            for part in pattern.split("."):
                obj = getattr(obj, part)
            return obj

        best, best_len = None, 0
        for _, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
                best, best_len = mod, len(mod)
        if best is None:
            raise ValueError("Could not auto-detect layers. Pass layer_pattern= explicitly.")
        return best

    def invalidate_cache(self):
        self._cached_heads = None

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def heads_per_layer(self) -> list[int]:
        if self._cached_heads is not None:
            return self._cached_heads

        heads = []
        for layer in self._layers:
            n = 1
            for attr_path in ["self_attn.num_heads", "attn.num_heads", "attention.self.num_attention_heads"]:
                try:
                    obj = layer
                    for part in attr_path.split("."):
                        obj = getattr(obj, part)
                    if isinstance(obj, int):
                        n = obj
                        break
                except AttributeError:
                    continue
            heads.append(n)
        self._cached_heads = heads
        return heads

    @property
    def params_per_layer(self) -> list[int]:
        return [sum(p.numel() for p in layer.parameters()) for layer in self._layers]

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def importance_scores(self) -> list[list[float]]:
        if self.importance_fn is not None and self.pruner is not None and _HAS_TP:
            return self._tp_importance()
        return self._l1_importance()

    def _tp_importance(self) -> list[list[float]]:
        scores = []
        for layer_idx, layer in enumerate(self._layers):
            n_heads = self.heads_per_layer[layer_idx]
            head_scores = [1.0] * n_heads
            scores.append(head_scores)
        return scores

    def _l1_importance(self) -> list[list[float]]:
        scores = []
        for layer_idx, layer in enumerate(self._layers):
            n_heads = self.heads_per_layer[layer_idx]
            out_proj = None
            for attr_path in ["self_attn.out_proj", "attn.c_proj", "attention.output.dense", "self_attn.o_proj"]:
                try:
                    obj = layer
                    for part in attr_path.split("."):
                        obj = getattr(obj, part)
                    out_proj = obj
                    break
                except AttributeError:
                    continue

            if out_proj is not None and hasattr(out_proj, "weight"):
                w = out_proj.weight.data
                d = w.shape[0]
                head_dim = d // n_heads if n_heads > 0 else d
                head_scores = []
                for h in range(n_heads):
                    s = h * head_dim
                    e = s + head_dim
                    if e <= w.shape[1]:
                        head_scores.append(round(w[:, s:e].abs().mean().item(), 6))
                    else:
                        head_scores.append(round(w.abs().mean().item(), 6))
                scores.append(head_scores)
            else:
                scores.append([1.0] * n_heads)
        return scores


class PruningCallback:

    def __init__(self, adapter: TorchPruningAdapter, recorder):
        self.adapter = adapter
        self.recorder = recorder
        self._step_counter = 0

    def on_prune_step(self, step: int, groups=None):
        from ..schema import ImportanceSnapshot, RewireEvent

        self.adapter.invalidate_cache()

        event = RewireEvent(
            step=step,
            event_type="prune_head",
            layer_idx=0,
            reason="torch-pruning step",
        )
        self.recorder.log_rewire(event)

        try:
            scores = self.adapter.importance_scores()
            self.recorder.log_importance(ImportanceSnapshot(step=step, scores=scores))
        except (NotImplementedError, Exception):
            pass

        self.recorder.log_snapshot(step, self.adapter)

    def wrap_pruner(self, pruner):
        original_step = pruner.step

        def wrapped_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self._step_counter += 1
            self.on_prune_step(step=self._step_counter)
            return result

        pruner.step = wrapped_step
