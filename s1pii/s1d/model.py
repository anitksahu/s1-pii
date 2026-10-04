"""Qwen3 backbone and the S1-D pointer head (no language-model head)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .packer import PackedWindow, SPECIAL_TOKENS

BACKBONES = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")


@dataclass
class PointerOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    hidden_states: torch.Tensor | None = None
    loss: torch.Tensor | None = None


class S1DModel(nn.Module):
    """A dense Qwen3 model plus projected decision/option pointer states."""
    def __init__(self, backbone: nn.Module, hidden_size: int | None = None):
        super().__init__()
        self.backbone = backbone
        hidden_size = hidden_size or backbone.config.hidden_size
        self.decision_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.option_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pointer_bias = nn.Parameter(torch.zeros(()))
        self.hidden_size = hidden_size

    @classmethod
    def from_config(cls, config):
        """Build a random local Qwen3Model, primarily for CPU tests."""
        from transformers import Qwen3Model
        # FlexAttention has no CPU autograd kernel. The CPU-only reference tests use an
        # explicit 4-D eager mask; every CUDA/production model uses a BlockMask.
        config._attn_implementation = "flex_attention" if torch.cuda.is_available() else "eager"
        return cls(Qwen3Model(config), config.hidden_size)

    @classmethod
    def from_pretrained(cls, model_id: str, revision: str, **kwargs):
        if model_id not in BACKBONES:
            raise ValueError(f"unsupported S1-D backbone {model_id!r}")
        if not revision:
            raise ValueError("a resolved Qwen revision is required")
        from transformers import AutoModel
        kwargs.pop("dtype", None)
        backbone = AutoModel.from_pretrained(model_id, revision=revision,
                                             attn_implementation="flex_attention", dtype=torch.bfloat16, **kwargs)
        if "qwen3" not in getattr(backbone.config, "model_type", "").lower():
            raise ValueError("S1-D requires a dense Qwen3 checkpoint")
        return cls(backbone)

    def enable_training_memory_features(self) -> None:
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.backbone.enable_input_require_grads()

    def forward(self, packed: PackedWindow, *, labels: torch.Tensor | None = None,
                output_hidden_states: bool = False) -> PointerOutput:
        device = next(self.parameters()).device
        # The BlockMask is intentionally supplied even with restarted position ids.
        attention_mask = packed.attention_mask
        implementation = getattr(self.backbone.config, "_attn_implementation", "")
        if device.type == "cpu" and implementation != "flex_attention":
            allowed = packed.dense_mask.to(device)
            attention_mask = torch.zeros_like(allowed, dtype=next(self.parameters()).dtype)
            attention_mask.masked_fill_(~allowed, torch.finfo(attention_mask.dtype).min)
            attention_mask = attention_mask[None, None]
        elif isinstance(attention_mask, torch.Tensor):
            raise ValueError("CUDA S1-D forward requires the prebuilt FlexAttention BlockMask")
        output = self.backbone(input_ids=packed.input_ids.unsqueeze(0).to(device),
                               position_ids=packed.position_ids.unsqueeze(0).to(device),
                               attention_mask=attention_mask,
                               output_hidden_states=output_hidden_states,
                               return_dict=True)
        hidden = output.last_hidden_state[0]
        decisions = self.decision_projection(hidden[packed.decide_indices.to(device)])
        option_states = self.option_projection(hidden[packed.option_indices.to(device)])
        if packed.layout == "kev":
            j = decisions.shape[0]
            if option_states.shape[0] % j:
                raise ValueError("Kev option rows must divide evenly over branches")
            options = option_states.view(j, -1, self.hidden_size)
            logits = torch.einsum("jd,jkd->jk", decisions, options)
        else:
            logits = decisions @ option_states.T
        logits = logits / math.sqrt(self.hidden_size) + self.pointer_bias
        loss = nn.functional.cross_entropy(logits, labels.to(device)) if labels is not None else None
        return PointerOutput(logits, logits.softmax(-1), hidden if output_hidden_states else None, loss)

    def forward_many(self, windows: list[PackedWindow]) -> list[PointerOutput]:
        """Evaluate equal-length windows in one backbone call with an explicit batched mask."""
        if not windows:
            return []
        lengths = {len(window.input_ids) for window in windows}
        if len(lengths) != 1:
            raise ValueError("forward_many requires one packed-length bucket")
        device = next(self.parameters()).device
        input_ids = torch.stack([window.input_ids for window in windows]).to(device)
        position_ids = torch.stack([window.position_ids for window in windows]).to(device)
        dense = torch.stack([window.dense_mask for window in windows]).to(device)
        implementation = getattr(self.backbone.config, "_attn_implementation", "")
        if device.type == "cpu" and implementation != "flex_attention":
            dtype = self.backbone.get_input_embeddings().weight.dtype
            attention_mask = torch.zeros_like(dense, dtype=dtype)
            attention_mask.masked_fill_(~dense, torch.finfo(dtype).min)
            attention_mask = attention_mask[:, None]
        else:
            from .packer import flex_batch_block_mask
            attention_mask = flex_batch_block_mask(dense, device=device)
        output = self.backbone(input_ids=input_ids, position_ids=position_ids,
                               attention_mask=attention_mask, return_dict=True)
        hidden = output.last_hidden_state
        decision_counts = [len(window.decide_indices) for window in windows]
        option_counts = [len(window.option_indices) for window in windows]
        decisions = self.decision_projection(torch.cat([
            hidden[index, window.decide_indices.to(device)] for index, window in enumerate(windows)
        ]))
        options = self.option_projection(torch.cat([
            hidden[index, window.option_indices.to(device)] for index, window in enumerate(windows)
        ]))
        decision_rows = decisions.split(decision_counts)
        option_rows = options.split(option_counts)
        results = []
        for window, decision, option in zip(windows, decision_rows, option_rows):
            if window.layout == "kev":
                option = option.view(decision.shape[0], -1, self.hidden_size)
                logits = torch.einsum("jd,jkd->jk", decision, option)
            else:
                logits = decision @ option.T
            logits = logits / math.sqrt(self.hidden_size) + self.pointer_bias
            results.append(PointerOutput(logits, logits.softmax(-1)))
        return results


def apply_lora(model: S1DModel, trainable_token_indices: list[int], rank: int = 32) -> S1DModel:
    """Attach PEFT LoRA to attention and MLP projections and train reserved rows."""
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        trainable_token_indices=list(trainable_token_indices),
    )
    model.backbone = get_peft_model(model.backbone, config)
    return model


def prepare_tokenizer(tokenizer, model: S1DModel | None = None) -> list[int]:
    """Allocate the six reserved rows and return their ids for PEFT."""
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    indices = [int(tokenizer.convert_tokens_to_ids(t)) for t in SPECIAL_TOKENS]
    if model is not None:
        rows = model.backbone.get_input_embeddings().num_embeddings
        if max(indices) >= rows:
            model.backbone.resize_token_embeddings(max(indices) + 1)
    return indices


def has_lm_head(model: nn.Module) -> bool:
    return any(name == "lm_head" or name.endswith(".lm_head") for name, _ in model.named_modules())


def merge_and_export(model: S1DModel, tokenizer, path: Path, manifest: dict | None = None) -> Path:
    """Merge LoRA for latency/deployment and save the backbone plus pointer head."""
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    if hasattr(model.backbone, "merge_and_unload"):
        model.backbone = model.backbone.merge_and_unload()
    model.backbone.save_pretrained(path / "backbone", safe_serialization=True)
    tokenizer.save_pretrained(path / "backbone")
    torch.save({"decision_projection": model.decision_projection.state_dict(),
                "option_projection": model.option_projection.state_dict(),
                "pointer_bias": model.pointer_bias.detach().cpu()}, path / "pointer.pt")
    (path / "manifest.json").write_text(__import__("json").dumps(manifest or {}, indent=2, sort_keys=True))
    return path
