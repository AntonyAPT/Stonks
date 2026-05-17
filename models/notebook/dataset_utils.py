"""Dataset helpers for PatchTST stock direction classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from labeling import LabelConfig, add_label_features, make_class_labels


@dataclass(frozen=True)
class WindowMetadata:
    """Human-readable metadata for a single sliding-window sample."""

    ticker: str
    context_start: pd.Timestamp
    context_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp


class StockWindowClassificationDataset(Dataset):
    """Sliding-window dataset for multi-day stock direction classification.

    `values_df` should contain the feature values used by the model. It can be a
    scaled/preprocessed DataFrame. `label_df` should contain the original close,
    high, and low values so label thresholds remain financially meaningful.

    Pass `ticker_industry` (dict mapping ticker → industry string) to enable
    industry embeddings. Pass `industry_to_id` from the training dataset when
    constructing val/test splits so all splits share the same mapping.
    """

    def __init__(
        self,
        values_df: pd.DataFrame,
        label_df: pd.DataFrame,
        target_columns: Sequence[str],
        context_length: int,
        forecast_horizon: int,
        label_config: LabelConfig,
        timestamp_column: str = "Date",
        ticker_column: str = "Ticker",
        ticker_industry: Optional[Dict[str, str]] = None,
        industry_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.target_columns = list(target_columns)
        self.context_length = int(context_length)
        self.forecast_horizon = int(forecast_horizon)
        self.timestamp_column = timestamp_column
        self.ticker_column = ticker_column
        self.label_config = label_config
        self.windows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, WindowMetadata]] = []

        required = {timestamp_column, ticker_column, *self.target_columns}
        missing_values = required.difference(values_df.columns)
        missing_labels = {timestamp_column, ticker_column, "Close", "High", "Low"}.difference(label_df.columns)
        if missing_values:
            raise ValueError(f"values_df is missing columns: {sorted(missing_values)}")
        if missing_labels:
            raise ValueError(f"label_df is missing columns: {sorted(missing_labels)}")

        # Build industry → integer mapping
        if ticker_industry:
            self.ticker_industry = ticker_industry
            if industry_to_id is not None:
                self.industry_to_id = industry_to_id
            else:
                all_industries = sorted(set(ticker_industry.values()))
                self.industry_to_id = {ind: i for i, ind in enumerate(all_industries)}
        else:
            self.ticker_industry = {}
            self.industry_to_id = {}
        self.num_industries = len(self.industry_to_id)

        label_df = add_label_features(label_df.copy(), vol_window=label_config.vol_window, ticker_column=ticker_column)
        values_df = values_df.copy()
        label_df = label_df.copy()
        values_df[timestamp_column] = pd.to_datetime(values_df[timestamp_column])
        label_df[timestamp_column] = pd.to_datetime(label_df[timestamp_column])

        for ticker, value_group in values_df.sort_values([ticker_column, timestamp_column]).groupby(ticker_column, sort=False):
            raw_group = label_df[label_df[ticker_column] == ticker].sort_values(timestamp_column)
            raw_features = raw_group[
                [
                    timestamp_column,
                    ticker_column,
                    "Close",
                    "rolling_vol",
                    "atr",
                ]
            ].rename(columns={"Close": "Close_raw"})
            merged = value_group.merge(
                raw_features,
                on=[timestamp_column, ticker_column],
                how="inner",
            )
            self._append_windows(str(ticker), merged)

    def _append_windows(self, ticker: str, frame: pd.DataFrame) -> None:
        values = frame[self.target_columns].astype("float32").to_numpy()
        close = frame["Close_raw"].astype("float64").to_numpy()
        rolling_vol = frame["rolling_vol"].astype("float64").to_numpy()
        atr = frame["atr"].astype("float64").to_numpy()
        dates = pd.to_datetime(frame[self.timestamp_column]).to_numpy()

        n_rows = len(frame)
        max_start = n_rows - self.context_length - self.forecast_horizon + 1
        if max_start <= 0:
            return

        for start in range(max_start):
            context_end = start + self.context_length
            forecast_end = context_end + self.forecast_horizon
            anchor_idx = context_end - 1

            labels = make_class_labels(
                future_close=close[context_end:forecast_end],
                past_close_t=close[anchor_idx],
                rule=self.label_config.rule,
                threshold=self.label_config.threshold,
                vol_window=self.label_config.vol_window,
                vol_k=self.label_config.vol_k,
                rolling_vol=rolling_vol[anchor_idx],
                atr=atr[anchor_idx],
            )

            metadata = WindowMetadata(
                ticker=ticker,
                context_start=pd.Timestamp(dates[start]),
                context_end=pd.Timestamp(dates[anchor_idx]),
                forecast_start=pd.Timestamp(dates[context_end]),
                forecast_end=pd.Timestamp(dates[forecast_end - 1]),
            )

            self.windows.append(
                (
                    values[start:context_end],
                    labels.astype("int64"),
                    values[context_end:forecast_end],
                    metadata,
                )
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        past_values, labels, future_values, meta = self.windows[idx]
        item = {
            "past_values": torch.tensor(past_values, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "future_values": torch.tensor(future_values, dtype=torch.float32),
        }
        if self.industry_to_id:
            industry = self.ticker_industry.get(meta.ticker, "Unknown")
            item["industry_id"] = torch.tensor(
                self.industry_to_id.get(industry, 0), dtype=torch.long
            )
        return item

    def metadata(self, idx: int) -> WindowMetadata:
        return self.windows[idx][3]

    def label_counts(self) -> np.ndarray:
        labels = np.concatenate([window[1] for window in self.windows]) if self.windows else np.array([], dtype=np.int64)
        return np.bincount(labels, minlength=3)


class ForecastClassificationDataset(Dataset):
    """Attach class labels to a ForecastDFDataset-like object.

    This wrapper is useful when experimenting directly with Granite TSFM's
    `ForecastDFDataset`. It expects each item to contain `past_values` and
    `future_values`, and computes labels from the close channel in those arrays.
    Use this only when those arrays are still in original price units.
    """

    def __init__(
        self,
        forecast_dataset: Dataset,
        close_channel_index: int,
        label_config: LabelConfig,
    ) -> None:
        self.forecast_dataset = forecast_dataset
        self.close_channel_index = int(close_channel_index)
        self.label_config = label_config

    def __len__(self) -> int:
        return len(self.forecast_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = dict(self.forecast_dataset[idx])
        past_values = torch.as_tensor(item["past_values"], dtype=torch.float32)
        future_values = torch.as_tensor(item["future_values"], dtype=torch.float32)
        past_close = float(past_values[-1, self.close_channel_index])
        future_close = future_values[:, self.close_channel_index].detach().cpu().numpy()
        labels = make_class_labels(
            future_close=future_close,
            past_close_t=past_close,
            rule=self.label_config.rule,
            threshold=self.label_config.threshold,
            vol_window=self.label_config.vol_window,
            vol_k=self.label_config.vol_k,
        )
        item["past_values"] = past_values
        item["future_values"] = future_values
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        item["class_labels"] = item["labels"]
        return item


def split_by_fraction(
    df: pd.DataFrame,
    train_frac: float,
    valid_frac: float,
    timestamp_column: str = "Date",
    ticker_column: str = "Ticker",
    context_length: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split each ticker, keeping context overlap for val/test."""

    train_frames: list[pd.DataFrame] = []
    valid_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for _, group in df.sort_values([ticker_column, timestamp_column]).groupby(ticker_column, sort=False):
        n = len(group)
        train_end = int(n * train_frac)
        valid_end = int(n * (train_frac + valid_frac))
        valid_start = max(0, train_end - context_length)
        test_start = max(0, valid_end - context_length)

        train_frames.append(group.iloc[:train_end])
        valid_frames.append(group.iloc[valid_start:valid_end])
        test_frames.append(group.iloc[test_start:])

    return (
        pd.concat(train_frames, ignore_index=True),
        pd.concat(valid_frames, ignore_index=True),
        pd.concat(test_frames, ignore_index=True),
    )


def compute_class_weights(dataset: StockWindowClassificationDataset, normalize: bool = True) -> torch.Tensor:
    """Compute inverse-frequency class weights for `CrossEntropyLoss`."""

    counts = dataset.label_counts().astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    weights = safe_counts.sum() / (len(safe_counts) * safe_counts)
    if normalize:
        weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def summarize_dataset(name: str, dataset: StockWindowClassificationDataset) -> pd.DataFrame:
    """Return a compact label distribution summary for notebook display."""

    counts = dataset.label_counts()
    total = counts.sum()
    pct = counts / max(total, 1)
    return pd.DataFrame(
        {
            "dataset": [name],
            "windows": [len(dataset)],
            "down": [int(counts[0])],
            "flat": [int(counts[1])],
            "up": [int(counts[2])],
            "down_pct": [pct[0]],
            "flat_pct": [pct[1]],
            "up_pct": [pct[2]],
        }
    )


def iter_tickers(df: pd.DataFrame, ticker_column: str = "Ticker") -> Iterable[str]:
    """Yield stable, sorted ticker symbols from a DataFrame."""

    return sorted(str(ticker) for ticker in df[ticker_column].dropna().unique())
