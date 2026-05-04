"""PatchTST classifier head for multi-day stock direction prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PatchTSTConfig, PatchTSTModel
from transformers.utils import ModelOutput


@dataclass
class PatchTSTClassifierOutput(ModelOutput):
    """Hugging Face Trainer-compatible output for the custom classifier."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class MultiDayClassificationHead(nn.Module):
    """Flatten PatchTST encoder states and emit `(batch, horizon, classes)` logits."""

    def __init__(
        self,
        d_model: int,
        num_patches: int,
        n_channels: int,
        horizon: int,
        n_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.n_classes = int(n_classes)
        self.dropout = nn.Dropout(dropout)

        # PatchTST output shape can vary slightly by transformers version
        # (`B,C,P,D` vs `B,P,C,D`). LazyLinear keeps this wrapper robust.
        del d_model, num_patches, n_channels
        self.projection = nn.LazyLinear(self.horizon * self.n_classes)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(encoder_hidden_states)
        flattened = hidden.reshape(hidden.shape[0], -1)
        logits = self.projection(flattened)
        return logits.view(-1, self.horizon, self.n_classes)


class PatchTSTClassifier(nn.Module):
    """PatchTST encoder plus a multi-day classification head.

    The forward signature intentionally accepts `past_values` and `labels` so it
    plugs into `transformers.Trainer` with the dataset in `dataset_utils.py`.

    Pass `num_industries > 0` to enable a learned sub-industry embedding that is
    concatenated onto the flattened encoder output before the final projection.
    The dataset must then supply an `industry_id` integer tensor per window.
    """

    def __init__(
        self,
        config: PatchTSTConfig,
        horizon: int,
        n_classes: int = 3,
        class_weights: Optional[torch.Tensor] = None,
        head_dropout: Optional[float] = None,
        num_industries: int = 0,
        industry_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.config = config
        self.horizon = int(horizon)
        self.n_classes = int(n_classes)
        self.patchtst = PatchTSTModel(config)

        patch_stride = getattr(config, "patch_stride", getattr(config, "patch_length", 1))
        num_patches = max(1, ((config.context_length - config.patch_length) // patch_stride) + 1)
        dropout = float(head_dropout if head_dropout is not None else getattr(config, "head_dropout", 0.2))
        n_channels = int(getattr(config, "num_input_channels", 1))
        self.classifier = MultiDayClassificationHead(
            d_model=int(config.d_model),
            num_patches=num_patches,
            n_channels=n_channels,
            horizon=self.horizon,
            n_classes=self.n_classes,
            dropout=dropout,
        )

        # Industry embedding: concatenated onto flattened encoder output.
        # A separate LazyLinear projects the combined vector to logits.
        self.num_industries = int(num_industries)
        if self.num_industries > 0:
            self.industry_embedding = nn.Embedding(num_industries, industry_embedding_dim)
            self.industry_projection = nn.LazyLinear(self.horizon * self.n_classes)
        else:
            self.industry_embedding = None
            self.industry_projection = None

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # Materialize LazyLinear weights with a dummy forward pass.
        # Required because transformers >= 4.56 calls `get_model_param_count`
        # (which invokes `.numel()` on every parameter) BEFORE the first real
        # forward pass, and uninitialized lazy params raise. Doing this in
        # __init__ keeps every caller (HF Trainer, manual loops, save/load)
        # working without each having to remember the warm-up step.
        self._materialize_lazy_params()

    def _materialize_lazy_params(self) -> None:
        n_channels = int(getattr(self.config, "num_input_channels", 1))
        dummy = torch.zeros(1, int(self.config.context_length), n_channels)
        dummy_industry = torch.zeros(1, dtype=torch.long) if self.num_industries > 0 else None
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self(past_values=dummy, industry_id=dummy_industry)
        if was_training:
            self.train()

    def _encoder_states(self, model_outputs: ModelOutput | Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if hasattr(model_outputs, "last_hidden_state") and model_outputs.last_hidden_state is not None:
            return model_outputs.last_hidden_state
        if hasattr(model_outputs, "hidden_states") and model_outputs.hidden_states is not None:
            return model_outputs.hidden_states[-1]
        if isinstance(model_outputs, tuple) and model_outputs:
            return model_outputs[0]
        raise RuntimeError("Could not find PatchTST encoder hidden states in model output.")

    def forward(
        self,
        past_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        industry_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PatchTSTClassifierOutput:
        del future_values
        label_tensor = labels if labels is not None else class_labels

        allowed_model_kwargs = {
            "past_observed_mask",
            "output_hidden_states",
            "output_attentions",
            "return_dict",
        }
        model_kwargs = {key: value for key, value in kwargs.items() if key in allowed_model_kwargs}
        outputs = self.patchtst(past_values=past_values, **model_kwargs)
        hidden_states = self._encoder_states(outputs)

        if self.industry_embedding is not None and industry_id is not None:
            flat = self.classifier.dropout(hidden_states).reshape(hidden_states.shape[0], -1)
            ind_emb = self.industry_embedding(industry_id.to(hidden_states.device))
            combined = torch.cat([flat, ind_emb], dim=-1)
            logits = self.industry_projection(combined).view(-1, self.horizon, self.n_classes)
        else:
            logits = self.classifier(hidden_states)

        loss = None
        if label_tensor is not None:
            label_tensor = label_tensor.to(logits.device).long()
            weights = None if self.class_weights is None else self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(logits.reshape(-1, self.n_classes), label_tensor.reshape(-1))

        return PatchTSTClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
