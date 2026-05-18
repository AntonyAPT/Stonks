"""OHLCV feature preprocessing for the technical PatchTST pipeline.

These helpers extract the scaling / column-normalisation steps that were
previously inline in the training notebook, making them importable by both
the training and evaluation notebooks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

OHLCV_FEATURE_COLUMNS: List[str] = ["Open", "High", "Low", "Close", "Volume"]

_CANONICAL_LOWER: Dict[str, str] = {
    "date": "Date",
    "ticker": "Ticker",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "sector": "Sector",
}


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename lowercase/alternate column names to canonical OHLCV names."""
    rename = {
        c: _CANONICAL_LOWER[c.strip().lower()]
        for c in df.columns
        if c.strip().lower() in _CANONICAL_LOWER
    }
    out = df.rename(columns=rename)
    if "Date" not in out.columns:
        raise ValueError(f"Expected a date column; got columns={list(df.columns)}")
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def scale_ohlcv_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str] = OHLCV_FEATURE_COLUMNS,
    ticker_column: str = "Ticker",
    timestamp_column: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a RobustScaler per ticker on the training split, then transform all splits.

    Volume is log1p-transformed before scaling.  Returns three DataFrames with
    the same columns as the inputs but with ``feature_columns`` replaced by
    their scaled values.  All non-feature columns (Ticker, Date, Sector, …)
    are preserved unchanged.
    """
    frames_train, frames_val, frames_test = [], [], []

    all_tickers = sorted(train_df[ticker_column].unique())

    for ticker in all_tickers:
        tr = train_df[train_df[ticker_column] == ticker].copy()
        va = val_df[val_df[ticker_column] == ticker].copy()
        te = test_df[test_df[ticker_column] == ticker].copy()

        if tr.empty:
            continue

        # Log-transform volume in-place before fitting the scaler.
        for df_slice in (tr, va, te):
            if "Volume" in feature_columns and "Volume" in df_slice.columns:
                df_slice["Volume"] = np.log1p(df_slice["Volume"].astype(float))

        scaler = RobustScaler()
        tr[feature_columns] = scaler.fit_transform(tr[feature_columns].astype("float32"))

        if not va.empty:
            va[feature_columns] = scaler.transform(va[feature_columns].astype("float32"))
        if not te.empty:
            te[feature_columns] = scaler.transform(te[feature_columns].astype("float32"))

        frames_train.append(tr)
        if not va.empty:
            frames_val.append(va)
        if not te.empty:
            frames_test.append(te)

    def _safe_concat(frames: list) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return _safe_concat(frames_train), _safe_concat(frames_val), _safe_concat(frames_test)
