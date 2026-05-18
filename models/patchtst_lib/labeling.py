"""Label-generation utilities for next-week stock direction classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

LabelRule = Literal["fixed_pct", "rolling_vol", "atr"]

DOWN_CLASS = 0
FLAT_CLASS = 1
UP_CLASS = 2
CLASS_NAMES = {DOWN_CLASS: "down", FLAT_CLASS: "flat", UP_CLASS: "up"}


@dataclass(frozen=True)
class LabelConfig:
    """Configuration for converting future closes into direction classes."""

    rule: LabelRule = "fixed_pct"
    threshold: float = 0.01
    vol_window: int = 21
    vol_k: float = 0.5


def _as_array(values: np.ndarray | pd.Series | list[float] | float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def fixed_pct_threshold(
    future_close: np.ndarray | pd.Series | list[float],
    past_close_t: np.ndarray | pd.Series | list[float] | float,
    threshold: float = 0.01,
) -> np.ndarray:
    """Classify future closes using a fixed percent move from the anchor close."""

    future = _as_array(future_close)
    anchor = np.expand_dims(_as_array(past_close_t), axis=-1) if np.ndim(past_close_t) > 0 else float(past_close_t)
    pct_change = (future - anchor) / np.maximum(np.abs(anchor), 1e-12)
    return np.where(pct_change > threshold, UP_CLASS, np.where(pct_change < -threshold, DOWN_CLASS, FLAT_CLASS))


def rolling_vol_threshold(
    future_close: np.ndarray | pd.Series | list[float],
    past_close_t: np.ndarray | pd.Series | list[float] | float,
    rolling_vol: np.ndarray | pd.Series | list[float] | float,
    vol_k: float = 0.5,
) -> np.ndarray:
    """Classify future closes using a rolling daily-return volatility threshold."""

    future = _as_array(future_close)
    anchor = np.expand_dims(_as_array(past_close_t), axis=-1) if np.ndim(past_close_t) > 0 else float(past_close_t)
    threshold = np.expand_dims(_as_array(rolling_vol), axis=-1) if np.ndim(rolling_vol) > 0 else float(rolling_vol)
    pct_change = (future - anchor) / np.maximum(np.abs(anchor), 1e-12)
    threshold = np.maximum(np.nan_to_num(threshold, nan=0.0), 0.0) * vol_k
    return np.where(pct_change > threshold, UP_CLASS, np.where(pct_change < -threshold, DOWN_CLASS, FLAT_CLASS))


def atr_threshold(
    future_close: np.ndarray | pd.Series | list[float],
    past_close_t: np.ndarray | pd.Series | list[float] | float,
    atr: np.ndarray | pd.Series | list[float] | float,
    vol_k: float = 0.5,
) -> np.ndarray:
    """Classify future closes using ATR as a close-normalized threshold."""

    anchor_raw = _as_array(past_close_t)
    future = _as_array(future_close)
    anchor = np.expand_dims(anchor_raw, axis=-1) if np.ndim(past_close_t) > 0 else float(past_close_t)
    atr_value = np.expand_dims(_as_array(atr), axis=-1) if np.ndim(atr) > 0 else float(atr)
    threshold = (np.maximum(np.nan_to_num(atr_value, nan=0.0), 0.0) / np.maximum(np.abs(anchor), 1e-12)) * vol_k
    pct_change = (future - anchor) / np.maximum(np.abs(anchor), 1e-12)
    return np.where(pct_change > threshold, UP_CLASS, np.where(pct_change < -threshold, DOWN_CLASS, FLAT_CLASS))


def compute_rolling_vol(close: pd.Series, window: int = 21) -> pd.Series:
    """Compute rolling standard deviation of daily close-to-close returns."""

    returns = close.astype(float).pct_change()
    return returns.rolling(window=window, min_periods=max(2, window // 2)).std()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Average True Range from high, low, and close series."""

    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=max(2, window // 2)).mean()


def make_class_labels(
    future_close: np.ndarray | pd.Series | list[float],
    past_close_t: np.ndarray | pd.Series | list[float] | float,
    rule: LabelRule = "fixed_pct",
    threshold: float = 0.01,
    vol_window: int = 21,
    vol_k: float = 0.5,
    rolling_vol: Optional[np.ndarray | pd.Series | list[float] | float] = None,
    atr: Optional[np.ndarray | pd.Series | list[float] | float] = None,
) -> np.ndarray:
    """Return class labels `{0, 1, 2}` per future day.

    Args:
        future_close: Future close values. Usually shaped `(horizon,)` or
            `(batch, horizon)`.
        past_close_t: Anchor close at the end of the context window.
        rule: Labeling rule: `fixed_pct`, `rolling_vol`, or `atr`.
        threshold: Fixed percent threshold used by `fixed_pct`.
        vol_window: Included for a consistent notebook API; rolling values
            should be precomputed for each anchor row.
        vol_k: Multiplier used by volatility-aware rules.
        rolling_vol: Precomputed rolling daily-return std for `rolling_vol`.
        atr: Precomputed ATR value for `atr`.

    Returns:
        Integer class array with the same leading shape as `future_close`.
    """

    del vol_window
    if rule == "fixed_pct":
        labels = fixed_pct_threshold(future_close, past_close_t, threshold)
    elif rule == "rolling_vol":
        if rolling_vol is None:
            raise ValueError("rolling_vol must be provided when rule='rolling_vol'.")
        labels = rolling_vol_threshold(future_close, past_close_t, rolling_vol, vol_k)
    elif rule == "atr":
        if atr is None:
            raise ValueError("atr must be provided when rule='atr'.")
        labels = atr_threshold(future_close, past_close_t, atr, vol_k)
    else:
        raise ValueError(f"Unsupported label rule: {rule!r}")

    return labels.astype(np.int64)


def add_label_features(df: pd.DataFrame, vol_window: int = 21, ticker_column: str = "Ticker") -> pd.DataFrame:
    """Add rolling-volatility and ATR helper columns per ticker."""

    required = {"High", "Low", "Close", ticker_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for label features: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    for _, group in df.sort_values([ticker_column, "Date"]).groupby(ticker_column, sort=False):
        group = group.copy()
        group["rolling_vol"] = compute_rolling_vol(group["Close"], window=vol_window)
        group["atr"] = compute_atr(group["High"], group["Low"], group["Close"], window=vol_window)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)
