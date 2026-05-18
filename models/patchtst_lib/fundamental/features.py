"""Fundamental feature engineering for the quarterly PatchTST pipeline.

Converts raw Compustat-style quarterly data (from quarterly_fundamentals_rows.csv)
into a normalised feature DataFrame suitable for PatchTST's ``past_values`` input.

Pipeline per ticker:
    1. Ticker harmonisation  (BRK.B → BRK-B)
    2. Sort by datadate
    3. Compute scale-invariant ratios
    4. Compute YoY / QoQ growth series
    5. Compute log-scaled level (log_atq)
    6. Winsorise each feature at 1st/99th percentile within ticker
    7. Z-score each feature within ticker
    8. Forward-fill any remaining NaN (PatchTST cannot tolerate NaN)

Cross-sectional rank features (rev_growth_rank_q) are computed across all tickers
sharing the same ``datadate`` so the model can see relative-to-peers signals.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature column list consumed by the dataset and PatchTST config.
# ---------------------------------------------------------------------------
FUND_FEATURE_COLUMNS: List[str] = [
    "net_margin",
    "roa",
    "roe",
    "debt_to_assets",
    "lt_debt_share",
    "rev_yoy",
    "rev_qoq",
    "niq_yoy",
    "eps_yoy",
    "log_atq",
    "rev_growth_rank_q",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_RAW_COLS = {"tic", "datadate", "niq", "epspxq", "atq", "ltq", "dlttq", "dlcq", "revtq"}


def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Element-wise division; returns NaN wherever denom <= 0."""
    return np.where(denom.abs() > 0, num / denom, np.nan)


def _winsorise_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip series to [lower, upper] quantile bounds (within-ticker)."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def _zscore_series(s: pd.Series) -> pd.Series:
    """Z-score a series; returns zeros if std == 0."""
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mean) / std


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def harmonise_tickers(df: pd.DataFrame, ticker_col: str = "tic") -> pd.DataFrame:
    """Replace dot notation with dash notation (BRK.B → BRK-B) to match OHLCV."""
    df = df.copy()
    df[ticker_col] = df[ticker_col].str.replace(".", "-", regex=False)
    return df


def load_raw_fundamentals(csv_path: str) -> pd.DataFrame:
    """Load and lightly validate the raw quarterly fundamentals CSV."""
    df = pd.read_csv(csv_path)
    missing = _REQUIRED_RAW_COLS - set(df.columns)
    if missing:
        raise ValueError(f"quarterly_fundamentals CSV missing columns: {sorted(missing)}")

    # Drop Supabase/Compustat plumbing columns that have no modelling value.
    for col in ("id", "gvkey", "fyearq", "fqtr"):
        if col in df.columns:
            df = df.drop(columns=[col])

    df = harmonise_tickers(df, ticker_col="tic")
    df["datadate"] = pd.to_datetime(df["datadate"])
    df = df.rename(columns={"tic": "Ticker"})
    return df


def _compute_per_ticker_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute all time-series features for a single ticker group.

    Input ``group`` is sorted by ``datadate`` and contains the raw Compustat
    columns: niq, epspxq, atq, ltq, dlttq, dlcq, revtq.
    """
    g = group.copy()

    # --- Ratios (row-wise, scale-invariant) ---------------------------------
    g["net_margin"]    = _safe_div(g["niq"],              g["revtq"])
    g["roa"]           = _safe_div(g["niq"],              g["atq"])
    equity             = g["atq"] - g["ltq"]
    g["roe"]           = _safe_div(g["niq"],              equity)
    total_debt         = g["dlttq"] + g["dlcq"]
    g["debt_to_assets"]= _safe_div(total_debt,            g["atq"])
    g["lt_debt_share"] = _safe_div(g["dlttq"],            total_debt)  # NaN if total_debt==0

    # --- Growth (per-ticker time series) ------------------------------------
    g["rev_yoy"]  = g["revtq"].pct_change(periods=4, fill_method=None)
    g["rev_qoq"]  = g["revtq"].pct_change(periods=1, fill_method=None)
    g["niq_yoy"]  = g["niq"].pct_change(periods=4, fill_method=None)
    g["eps_yoy"]  = g["epspxq"].pct_change(periods=4, fill_method=None)

    # --- Log level ----------------------------------------------------------
    g["log_atq"] = np.log(g["atq"].clip(lower=1.0))

    return g


def _normalise_ticker(group: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Winsorise then z-score each feature column within a single ticker."""
    g = group.copy()
    for col in feature_cols:
        if col not in g.columns:
            continue
        s = g[col].astype(float)
        if s.notna().sum() < 2:
            continue
        s = _winsorise_series(s)
        s = _zscore_series(s)
        g[col] = s
    return g


def build_feature_df(
    raw_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Main entry point: raw fundamentals DataFrame → normalised feature DataFrame.

    Parameters
    ----------
    raw_df:
        Output of ``load_raw_fundamentals``, with columns
        ``Ticker, datadate, niq, epspxq, atq, ltq, dlttq, dlcq, revtq``.
    feature_columns:
        Feature columns to include in the output.  Defaults to
        ``FUND_FEATURE_COLUMNS``.

    Returns
    -------
    pd.DataFrame with columns ``[Ticker, datadate] + feature_columns``,
    sorted by ``[Ticker, datadate]``.  All feature values are float32.
    """
    if feature_columns is None:
        feature_columns = FUND_FEATURE_COLUMNS

    # Step 1: compute raw per-ticker features.
    ticker_frames = []
    for ticker, group in raw_df.sort_values(["Ticker", "datadate"]).groupby("Ticker", sort=False):
        group = group.sort_values("datadate").reset_index(drop=True)
        group = _compute_per_ticker_features(group)
        ticker_frames.append(group)

    df = pd.concat(ticker_frames, ignore_index=True)

    # Step 2: cross-sectional rank of rev_yoy per datadate.
    # Rank ascending so rank=1 means worst rev growth, high rank means best.
    # Normalise to [0, 1] so the scale is comparable across quarters with
    # different universe sizes.
    if "rev_growth_rank_q" in feature_columns:
        df["rev_growth_rank_q"] = (
            df.groupby("datadate")["rev_yoy"]
            .rank(method="average", na_option="keep")
            .div(df.groupby("datadate")["rev_yoy"].transform("count"))
        )

    # Step 3: winsorise + z-score per ticker.
    # Only operate on numeric feature columns that exist.
    numeric_feats = [c for c in feature_columns if c in df.columns]
    normalised_frames = []
    for _, group in df.groupby("Ticker", sort=False):
        group = _normalise_ticker(group, numeric_feats)
        normalised_frames.append(group)

    df = pd.concat(normalised_frames, ignore_index=True)

    # Step 4: fill NaN within each ticker (PatchTST cannot tolerate NaN).
    # Forward-fill first (use most recent valid observation), then back-fill
    # (handles leading NaN at the start of a ticker's history), then fill any
    # remaining NaN with 0.0 (z-scored mean = "no information available").
    # This handles financial-sector tickers (e.g. banks) where revtq is not
    # reported, leaving revenue-derived features NaN for the entire series.
    df = df.sort_values(["Ticker", "datadate"])
    for col in numeric_feats:
        df[col] = df.groupby("Ticker")[col].transform(lambda s: s.ffill().bfill().fillna(0.0))

    # Cast to float32.
    for col in numeric_feats:
        df[col] = df[col].astype("float32")

    keep_cols = ["Ticker", "datadate"] + [c for c in feature_columns if c in df.columns]
    return df[keep_cols].sort_values(["Ticker", "datadate"]).reset_index(drop=True)
