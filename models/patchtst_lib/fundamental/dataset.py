"""Quarterly fundamental dataset for the PatchTST fundamental pipeline.

Each sample is a sliding window of ``context_length`` consecutive quarterly
fundamental feature vectors for one ticker, paired with a 3-class direction
label derived from the stock's return over the following quarter.

Key design decisions
---------------------
Publication-lag guard
    Compustat ``datadate`` is the fiscal quarter *end* date, but the 10-Q /
    10-K isn't filed until several weeks later.  To prevent lookahead bias,
    the model's "decision date" is defined as
        ``anchor_date = datadate + publish_lag_days``   (default 45 days)
    The entry close is the first OHLCV close on or after ``anchor_date``; the
    exit close is the first OHLCV close on or after
        ``anchor_date + forecast_lag_days``             (default 63 trading days
                                                         ≈ one quarter).

Fundamentals are the only model input
    ``prices_df`` (from ``historic_data_rows.csv``) is used *only* to compute
    the label — it never appears in ``past_values``.

NaN handling
    Windows that still contain NaN in ``past_values`` after forward-filling
    are silently dropped so PatchTST never receives NaN input.  A yield
    summary is available via ``skipped_tickers`` and ``window_counts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from patchtst_lib.labeling import LabelConfig, make_class_labels


@dataclass(frozen=True)
class QuarterlyWindowMetadata:
    """Human-readable metadata for a single quarterly window."""

    ticker: str
    context_start_quarter: pd.Timestamp
    context_end_quarter: pd.Timestamp
    decision_date: pd.Timestamp    # datadate + publish_lag
    forecast_end_date: pd.Timestamp


class QuarterlyFundamentalDataset(Dataset):
    """Sliding-window dataset over quarterly fundamental features.

    Parameters
    ----------
    features_df:
        Output of ``patchtst_lib.fundamental.features.build_feature_df`` —
        columns ``[Ticker, datadate] + feature_columns``.
    prices_df:
        Long-form OHLCV DataFrame with at least ``[Date, Ticker, Close]``.
        Used *only* for label computation; never returned by ``__getitem__``.
    feature_columns:
        List of feature column names to use as model input channels.
    context_length:
        Number of consecutive quarters in each context window (default 12).
    label_config:
        Labelling rule (fixed_pct / rolling_vol / atr).
    publish_lag_days:
        Calendar days added to each anchor ``datadate`` before reading the
        entry close price.  Default 45 (conservative 10-Q filing window).
    forecast_lag_days:
        Calendar days from the entry date to the exit date.  Default 63
        (≈ one quarter in trading days expressed as calendar days).
    ticker_column:
        Column name for tickers in both DataFrames.
    date_column:
        Column name for dates in ``prices_df``.
    close_column:
        Column name for close prices in ``prices_df``.
    ticker_industry:
        Optional dict mapping ticker → industry string for industry embeddings.
    industry_to_id:
        Optional dict mapping industry string → integer id.  Pass in when
        constructing val/test splits so all splits share the same mapping.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        feature_columns: Sequence[str],
        context_length: int = 12,
        label_config: Optional[LabelConfig] = None,
        publish_lag_days: int = 45,
        forecast_lag_days: int = 63,
        ticker_column: str = "Ticker",
        date_column: str = "Date",
        close_column: str = "Close",
        ticker_industry: Optional[Dict[str, str]] = None,
        industry_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.context_length = int(context_length)
        self.label_config = label_config or LabelConfig()
        self.publish_lag = timedelta(days=int(publish_lag_days))
        self.forecast_lag = timedelta(days=int(forecast_lag_days))
        self.ticker_column = ticker_column

        # Build industry → integer mapping (mirrors StockWindowClassificationDataset).
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

        # Pre-build an asof-compatible price index per ticker.
        prices_df = prices_df.copy()
        prices_df[date_column] = pd.to_datetime(prices_df[date_column])
        # Build (ticker, date) → close lookup (price_lookup) for O(1) access.
        price_lookup: Dict[str, pd.Series] = {}
        for ticker, grp in prices_df.groupby(ticker_column, sort=False):
            grp = grp.sort_values(date_column).set_index(date_column)
            price_lookup[str(ticker)] = grp[close_column].astype(float)

        features_df = features_df.copy()
        features_df["datadate"] = pd.to_datetime(features_df["datadate"])

        self.windows: List[
            Tuple[np.ndarray, np.ndarray, QuarterlyWindowMetadata]
        ] = []
        self.window_counts: Dict[str, int] = {}
        self.skipped_tickers: List[str] = []

        for ticker, group in features_df.sort_values(
            [ticker_column, "datadate"]
        ).groupby(ticker_column, sort=False):
            ticker = str(ticker)
            group = group.sort_values("datadate").reset_index(drop=True)
            price_series = price_lookup.get(ticker)
            if price_series is None:
                self.skipped_tickers.append(ticker)
                continue
            n = self._append_windows(ticker, group, price_series)
            self.window_counts[ticker] = n
            if n == 0:
                self.skipped_tickers.append(ticker)

    # ------------------------------------------------------------------
    def _asof_close(self, price_series: pd.Series, target_date: pd.Timestamp) -> Optional[float]:
        """Return the first closing price on or after ``target_date``."""
        valid = price_series[price_series.index >= target_date]
        if valid.empty:
            return None
        return float(valid.iloc[0])

    def _append_windows(
        self, ticker: str, group: pd.DataFrame, price_series: pd.Series
    ) -> int:
        """Build all sliding windows for one ticker. Returns count added."""
        n_rows = len(group)
        max_start = n_rows - self.context_length
        if max_start <= 0:
            return 0

        values = group[self.feature_columns].astype("float32").to_numpy()
        datadates = group["datadate"].to_numpy()

        added = 0
        for start in range(max_start):
            context_end = start + self.context_length - 1  # inclusive
            anchor_quarter = pd.Timestamp(datadates[context_end])

            decision_date = anchor_quarter + self.publish_lag
            forecast_end_date = decision_date + self.forecast_lag

            entry_close = self._asof_close(price_series, decision_date)
            exit_close = self._asof_close(price_series, forecast_end_date)

            if entry_close is None or exit_close is None or entry_close <= 0:
                continue

            window_values = values[start : context_end + 1]  # (context_length, n_features)

            # Drop windows with any NaN after forward-fill in features.py.
            if np.isnan(window_values).any():
                continue

            label_arr = make_class_labels(
                future_close=np.array([exit_close]),
                past_close_t=entry_close,
                rule=self.label_config.rule,
                threshold=self.label_config.threshold,
                vol_window=self.label_config.vol_window,
                vol_k=self.label_config.vol_k,
            )  # shape (1,)

            metadata = QuarterlyWindowMetadata(
                ticker=ticker,
                context_start_quarter=pd.Timestamp(datadates[start]),
                context_end_quarter=anchor_quarter,
                decision_date=decision_date,
                forecast_end_date=forecast_end_date,
            )

            self.windows.append((window_values, label_arr.astype("int64"), metadata))
            added += 1

        return added

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        past_values, labels, meta = self.windows[idx]
        item: Dict[str, torch.Tensor] = {
            "past_values": torch.tensor(past_values, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "class_labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.industry_to_id:
            industry = self.ticker_industry.get(meta.ticker, "Unknown")
            item["industry_id"] = torch.tensor(
                self.industry_to_id.get(industry, 0), dtype=torch.long
            )
        return item

    def metadata(self, idx: int) -> QuarterlyWindowMetadata:
        return self.windows[idx][2]

    def label_counts(self) -> np.ndarray:
        if not self.windows:
            return np.array([], dtype=np.int64)
        labels = np.concatenate([w[1] for w in self.windows])
        return np.bincount(labels, minlength=3)

    def yield_summary(self) -> pd.DataFrame:
        """Return a per-ticker window-count DataFrame for diagnostics."""
        rows = [{"ticker": t, "windows": n} for t, n in sorted(self.window_counts.items())]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset utilities (mirror of patchtst_lib.technical.dataset)
# ---------------------------------------------------------------------------

def split_fundamentals_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    date_column: str = "datadate",
    ticker_column: str = "Ticker",
    context_length: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a features DataFrame chronologically by absolute date cutoffs.

    Validation and test splits receive ``context_length`` quarters of overlap
    with the preceding split so the first window of each split is fully
    populated.

    Parameters
    ----------
    df:
        Long-form features DataFrame with ``[ticker_column, date_column, ...]``.
    train_end:
        Inclusive upper bound for training data (e.g. ``"2019-12-31"``).
    val_end:
        Inclusive upper bound for validation data (e.g. ``"2021-12-31"``).
    context_length:
        Number of quarters in each context window; used to compute overlap.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train_frames, val_frames, test_frames = [], [], []

    for _, group in df.sort_values([ticker_column, date_column]).groupby(
        ticker_column, sort=False
    ):
        group = group.sort_values(date_column)
        dates = group[date_column]

        # Training: everything up to train_end.
        tr = group[dates <= train_end_dt]
        if tr.empty:
            continue
        train_frames.append(tr)

        # Validation: overlap by context_length quarters, up to val_end.
        val_start = tr.iloc[max(0, len(tr) - context_length)][date_column]
        va = group[(dates >= val_start) & (dates <= val_end_dt)]
        if not va.empty:
            val_frames.append(va)

        # Test: overlap by context_length quarters, from val_end onward.
        test_start_idx = group[dates <= val_end_dt]
        if not test_start_idx.empty:
            test_overlap_start = test_start_idx.iloc[
                max(0, len(test_start_idx) - context_length)
            ][date_column]
            te = group[dates >= test_overlap_start]
            if not te.empty:
                test_frames.append(te)

    def _safe_concat(frames: list) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=df.columns)

    return _safe_concat(train_frames), _safe_concat(val_frames), _safe_concat(test_frames)


def compute_fundamental_class_weights(
    dataset: QuarterlyFundamentalDataset, normalize: bool = True
) -> torch.Tensor:
    """Inverse-frequency class weights for ``CrossEntropyLoss``."""
    counts = dataset.label_counts().astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    weights = safe_counts.sum() / (len(safe_counts) * safe_counts)
    if normalize:
        weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def summarize_fundamental_dataset(
    name: str, dataset: QuarterlyFundamentalDataset
) -> pd.DataFrame:
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
