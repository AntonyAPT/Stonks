"""PatchTST configuration factory for the quarterly fundamental pipeline.

The fundamental model is intentionally small relative to the technical model:
- Context: 12 quarters (vs 128 days technical)
- Patches: (12 - 2) / 1 + 1 = 11 patches (vs ~16 technical)
- d_model: 32 (vs 128 technical) — ~4,700 windows demand a compact model
- Depth: 2 layers (vs 3 technical)

Refer to ``classification_head.PatchTSTClassifier`` for the full model
definition; this module only creates the ``PatchTSTConfig`` that drives it.
"""

from __future__ import annotations

from typing import Optional

from transformers import PatchTSTConfig

from patchtst_lib.fundamental.features import FUND_FEATURE_COLUMNS


def make_fundamental_patchtst_config(
    context_length: int = 12,
    patch_length: int = 2,
    patch_stride: int = 1,
    num_input_channels: Optional[int] = None,
    d_model: int = 32,
    num_attention_heads: int = 4,
    num_hidden_layers: int = 2,
    ffn_dim: Optional[int] = None,
    head_dropout: float = 0.3,
    attention_dropout: float = 0.2,
    dropout: float = 0.2,
    positional_encoding_type: str = "sincos",
    use_cls_token: bool = False,
) -> PatchTSTConfig:
    """Return a ``PatchTSTConfig`` tuned for quarterly fundamental data.

    Parameters
    ----------
    context_length:
        Number of quarters in each context window.  Default 12 (three years).
    patch_length:
        Number of time steps per patch.  Default 2 (two quarters per patch).
    patch_stride:
        Stride between consecutive patches.  Default 1 (fully overlapping).
    num_input_channels:
        Number of fundamental feature channels.  Defaults to
        ``len(FUND_FEATURE_COLUMNS)`` (= 11).
    d_model:
        Transformer hidden dimension.  Keep at 32 for the initial run with
        ~4,700 windows; bump to 64 only if val-loss plateaus high.
    num_attention_heads:
        Number of attention heads.  Must divide ``d_model``.
    num_hidden_layers:
        Transformer encoder depth.
    ffn_dim:
        Feed-forward network dimension.  Defaults to ``4 * d_model``.
    head_dropout:
        Dropout applied inside ``MultiDayClassificationHead``.
    attention_dropout:
        Dropout applied to attention weights.
    dropout:
        General dropout rate.
    positional_encoding_type:
        Type of positional encoding — ``"sincos"`` or ``"random"``.
    use_cls_token:
        Whether to prepend a [CLS] token.

    Returns
    -------
    ``PatchTSTConfig`` — pass directly to ``PatchTSTClassifier``.
    """
    if num_input_channels is None:
        num_input_channels = len(FUND_FEATURE_COLUMNS)
    if ffn_dim is None:
        ffn_dim = 4 * d_model

    # Sanity check: num_attention_heads must divide d_model.
    if d_model % num_attention_heads != 0:
        raise ValueError(
            f"d_model={d_model} must be divisible by num_attention_heads={num_attention_heads}."
        )

    return PatchTSTConfig(
        num_input_channels=num_input_channels,
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        prediction_length=1,         # single-quarter forecast horizon
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        ffn_dim=ffn_dim,
        head_dropout=head_dropout,
        attention_dropout=attention_dropout,
        dropout=dropout,
        positional_encoding_type=positional_encoding_type,
        use_cls_token=use_cls_token,
        # Channel-independent mode (PatchTST default): each feature is
        # processed independently through the same transformer.
        channel_attention=False,
    )
