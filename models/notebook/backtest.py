"""Paper-trading backtest module for PatchTST stock direction predictions.

Usage
-----
    from backtest import build_prediction_df, run_backtest, summarize_results

    # Build a flat DataFrame from model outputs (logits, labels) and price data.
    pred_df = build_prediction_df(logits, labels, test_ds, raw_df)

    # Run the paper-trade backtest on 1 year of predictions.
    results = run_backtest(pred_df, starting_capital=1000, top_n=10)

    # Pretty-print summary and return the summary dict.
    summary = summarize_results(results)

Lookahead-bias contract
-----------------------
- Each row's ``trade_date`` equals ``context_end`` — the last day of
  the context window the model *actually saw* before making a prediction.
- ``forward_return`` = (close[forecast_start] − close[context_end])
  / close[context_end], where ``forecast_start`` is the *next* trading
  day after ``context_end``.  The model never sees this price.
- ``select_top_confident_up`` operates on a single day's predictions and
  never looks at returns before selecting which stocks to hold.
- ``run_backtest`` applies ``forward_return`` *after* the selection step.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax — avoids the scipy dependency."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# Matches labeling.py constants (0=down, 1=flat, 2=up).
CLASS_NAMES = {0: "down", 1: "flat", 2: "up"}
UP_CLASS = 2
FLAT_CLASS = 1


# ---------------------------------------------------------------------------
# 1. Build prediction DataFrame
# ---------------------------------------------------------------------------

def build_prediction_df(
    logits: np.ndarray,
    labels: np.ndarray,
    dataset,
    price_df: pd.DataFrame,
    *,
    day_idx: int = 0,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    close_col: str = "Close",
) -> pd.DataFrame:
    """Convert raw model outputs into a flat per-window prediction DataFrame.

    Parameters
    ----------
    logits:
        Shape ``(N, horizon, 3)``.  Raw logits from the model (before softmax).
    labels:
        Shape ``(N, horizon)``.  Ground-truth class indices (0/1/2).
    dataset:
        A ``StockWindowClassificationDataset`` that supports ``.metadata(i)``
        returning a ``WindowMetadata`` object.
    price_df:
        Raw price DataFrame containing at least [date_col, ticker_col, close_col].
        Used *only* to compute ``forward_return``; no future prices leak into
        the selection step.
    day_idx:
        Which forecast day to evaluate.  0 = Day 1 (nearest, default).
    date_col, ticker_col, close_col:
        Column names in ``price_df``.

    Returns
    -------
    pd.DataFrame with columns:

    - ``trade_date``        — ``context_end`` (decision date, last context day)
    - ``forecast_date``     — ``forecast_start`` (first forecast day / entry day)
    - ``ticker``
    - ``predicted_class``   — 0 / 1 / 2
    - ``predicted_direction`` — "down" / "flat" / "up"
    - ``confidence``        — softmax probability assigned to the predicted class
    - ``actual_class``      — 0 / 1 / 2
    - ``actual_direction``  — "down" / "flat" / "up"
    - ``forward_return``    — (close[forecast_start] − close[context_end]) /
                              close[context_end].  NaN when prices are missing.
    """
    # Pre-build a (ticker, date) → close price lookup for O(1) access.
    price_df = price_df.copy()
    price_df[date_col] = pd.to_datetime(price_df[date_col])
    price_lookup: dict[tuple, float] = (
        price_df.set_index([ticker_col, date_col])[close_col]
        .to_dict()
    )

    # Compute softmax probabilities once over the full batch.
    probs = _softmax(logits, axis=-1)  # (N, horizon, 3)

    rows = []
    for i in range(len(dataset)):
        meta = dataset.metadata(i)

        pred_class = int(np.argmax(logits[i, day_idx]))
        actual_class = int(labels[i, day_idx])
        confidence = float(probs[i, day_idx, pred_class])

        # forward_return: realized return the model is predicting the direction of.
        # Uses context_end close (known at decision time) and forecast_start close
        # (realized after the trade is entered — no lookahead).
        close_anchor = price_lookup.get((meta.ticker, meta.context_end))
        close_forecast = price_lookup.get((meta.ticker, meta.forecast_start))

        if (
            close_anchor is not None
            and close_forecast is not None
            and close_anchor > 0
        ):
            forward_return = (close_forecast - close_anchor) / close_anchor
        else:
            forward_return = np.nan

        rows.append(
            {
                "trade_date": meta.context_end,
                "forecast_date": meta.forecast_start,
                "ticker": meta.ticker,
                "predicted_class": pred_class,
                "predicted_direction": CLASS_NAMES[pred_class],
                "confidence": confidence,
                "actual_class": actual_class,
                "actual_direction": CLASS_NAMES[actual_class],
                "forward_return": forward_return,
            }
        )

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df.sort_values(["trade_date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Direction accuracy (excluding flat predictions and flat actuals)
# ---------------------------------------------------------------------------

def compute_direction_accuracy(df: pd.DataFrame) -> dict:
    """Compute up/down direction accuracy, ignoring flat predictions and actuals.

    Rows are excluded from the calculation when *either* the predicted direction
    or the actual direction is "flat".  A prediction is "correct" when both the
    predicted and actual non-flat directions match.

    Parameters
    ----------
    df:
        DataFrame produced by ``build_prediction_df`` (must contain
        ``predicted_direction`` and ``actual_direction``).

    Returns
    -------
    dict with keys:
    - ``total_trades``   — rows evaluated (non-flat on both sides)
    - ``correct``        — correct non-flat predictions
    - ``incorrect``      — incorrect non-flat predictions
    - ``accuracy``       — correct / total_trades, or None if total_trades == 0
    """
    mask = (df["predicted_direction"] != "flat") & (df["actual_direction"] != "flat")
    filtered = df[mask]

    total = len(filtered)
    if total == 0:
        return {"total_trades": 0, "correct": 0, "incorrect": 0, "accuracy": None}

    correct = int((filtered["predicted_direction"] == filtered["actual_direction"]).sum())
    incorrect = total - correct

    return {
        "total_trades": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": correct / total,
    }


# ---------------------------------------------------------------------------
# 3. Confidence ranking — select top-N predicted-up stocks per day
# ---------------------------------------------------------------------------

def select_top_confident_up(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top-N predicted-up stocks ranked by confidence (descending).

    This function is meant to be called on a *single trading day's* predictions
    (i.e., rows where ``trade_date`` == some constant).  Passing a multi-day
    DataFrame will silently process all rows together, which is not the intended
    use case for the backtest loop.

    Selection rules:
    - Keep only rows where ``predicted_direction == "up"``
    - Rank by ``confidence`` descending
    - Return at most ``top_n`` rows (fewer if not enough qualify)

    Parameters
    ----------
    df:
        One trading day's prediction DataFrame.
    top_n:
        Maximum number of stocks to select.

    Returns
    -------
    pd.DataFrame — subset of ``df``, sorted by confidence descending.
    """
    up_preds = df[df["predicted_direction"] == "up"].copy()
    return up_preds.nlargest(top_n, "confidence", keep="first")


# ---------------------------------------------------------------------------
# 4. Paper-trading backtest
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    starting_capital: float = 1000.0,
    top_n: int = 10,
    benchmark_prices: Optional[pd.Series] = None,
) -> dict:
    """Simulate a daily paper-trade strategy on the prediction DataFrame.

    Strategy
    --------
    For each trading day (grouped by ``trade_date``):
    1. Select the top ``top_n`` predicted-up stocks by confidence.
    2. Allocate capital equally across the selected stocks.
    3. Apply each stock's ``forward_return`` to compute the day's P&L.
    4. Roll the updated portfolio value into the next day.

    If fewer than ``top_n`` stocks are predicted "up" on a given day, the
    available stocks are equal-weighted (capital stays fully invested).
    If zero stocks are predicted "up", capital sits in cash (0% return) for
    that day.

    Lookahead-bias check
    --------------------
    ``forward_return`` is applied *after* the selection step.  The selection
    uses only ``predicted_direction`` and ``confidence``, both of which are
    derived from model outputs that do not touch the forecast period's prices.

    Parameters
    ----------
    df:
        Prediction DataFrame from ``build_prediction_df``.  Rows with NaN
        ``forward_return`` are dropped before processing.
    starting_capital:
        Initial portfolio value in dollars.
    top_n:
        Number of top-confidence "up" stocks to hold each day.
    benchmark_prices:
        Optional pd.Series indexed by date with a benchmark price (e.g., SPY
        daily closes).  If supplied, benchmark returns are computed over the
        same date range and included in results.

    Returns
    -------
    dict with keys:

    - ``daily_equity``              — pd.Series(date → portfolio_value)
    - ``final_value``               — float
    - ``total_return_pct``          — float
    - ``trading_days``              — int
    - ``avg_positions``             — float (avg # of stocks held per day)
    - ``direction_accuracy_all``    — dict from ``compute_direction_accuracy``
    - ``direction_accuracy_selected`` — dict (only the selected rows)
    - ``benchmark_return_pct``      — float or None (only if benchmark_prices given)
    """
    # Drop rows where we cannot compute a realized return.
    df = df.dropna(subset=["forward_return"]).copy()
    if df.empty:
        raise ValueError("No rows with valid forward_return — cannot run backtest.")

    portfolio_value = starting_capital
    daily_records: list[dict] = []
    positions_per_day: list[int] = []
    selected_frames: list[pd.DataFrame] = []

    # Iterate in chronological order — no future data can leak in.
    for trade_date in sorted(df["trade_date"].unique()):
        day_preds = df[df["trade_date"] == trade_date]
        selected = select_top_confident_up(day_preds, top_n)
        n_positions = len(selected)
        positions_per_day.append(n_positions)

        if n_positions > 0:
            # Equal weight: each stock gets 1/n_positions of the portfolio.
            weight = 1.0 / n_positions
            day_return = float((selected["forward_return"] * weight).sum())
            portfolio_value = portfolio_value * (1.0 + day_return)
            selected_frames.append(selected)

        # Record *after* the day's return is applied.
        daily_records.append(
            {
                "date": trade_date,
                "portfolio_value": portfolio_value,
                "n_positions": n_positions,
            }
        )

    daily_df = pd.DataFrame(daily_records)
    daily_equity = daily_df.set_index("date")["portfolio_value"]

    # Direction accuracy over ALL non-flat predictions in the dataset.
    acc_all = compute_direction_accuracy(df)

    # Direction accuracy only for the rows that were actually traded.
    if selected_frames:
        traded = pd.concat(selected_frames, ignore_index=True)
        acc_selected = compute_direction_accuracy(traded)
    else:
        acc_selected = {"total_trades": 0, "correct": 0, "incorrect": 0, "accuracy": None}

    # Optional benchmark: compute buy-and-hold return over the backtest window.
    benchmark_return_pct = None
    if benchmark_prices is not None:
        bench = benchmark_prices.sort_index()
        start_date = daily_equity.index.min()
        end_date = daily_equity.index.max()
        bench_window = bench.loc[
            (bench.index >= start_date) & (bench.index <= end_date)
        ]
        if len(bench_window) >= 2:
            benchmark_return_pct = (
                bench_window.iloc[-1] / bench_window.iloc[0] - 1
            ) * 100

    if selected_frames:
        traded_returns = pd.concat(selected_frames, ignore_index=True)["forward_return"]
        avg_forward_return_selected = float(traded_returns.mean())
    else:
        avg_forward_return_selected = float("nan")

    return {
        "daily_equity": daily_equity,
        "final_value": portfolio_value,
        "total_return_pct": (portfolio_value / starting_capital - 1) * 100,
        "trading_days": len(daily_records),
        "avg_positions": float(np.mean(positions_per_day)) if positions_per_day else 0.0,
        "avg_forward_return_selected": avg_forward_return_selected,
        "direction_accuracy_all": acc_all,
        "direction_accuracy_selected": acc_selected,
        "benchmark_return_pct": benchmark_return_pct,
    }


# ---------------------------------------------------------------------------
# 5. Results summary
# ---------------------------------------------------------------------------

def summarize_results(backtest_results: dict) -> dict:
    """Print a formatted backtest summary and return the summary as a dict.

    Parameters
    ----------
    backtest_results:
        The dict returned by ``run_backtest``.

    Returns
    -------
    dict — same key/value pairs printed to stdout, with numeric types preserved.
    """
    r = backtest_results
    eq = r["daily_equity"]

    acc_all = r["direction_accuracy_all"]
    acc_sel = r["direction_accuracy_selected"]

    def fmt_acc(acc_dict: dict) -> str:
        if acc_dict["accuracy"] is None:
            return "N/A (0 non-flat trades)"
        return (
            f"{acc_dict['accuracy']:.2%}  "
            f"({acc_dict['correct']}/{acc_dict['total_trades']} correct)"
        )

    lines = [
        "",
        "=" * 54,
        "  BACKTEST RESULTS",
        "=" * 54,
        f"  {'Starting Capital':<36} ${1000.0:>10,.2f}",
        f"  {'Final Portfolio Value':<36} ${r['final_value']:>10,.2f}",
        f"  {'Total Return':<36} {r['total_return_pct']:>+10.2f}%",
    ]

    if r["benchmark_return_pct"] is not None:
        lines.append(
            f"  {'Benchmark Return (buy & hold)':<36} "
            f"{r['benchmark_return_pct']:>+10.2f}%"
        )

    afr = r["avg_forward_return_selected"]
    afr_str = f"{afr:>+10.4%}" if not np.isnan(afr) else "       N/A"

    lines += [
        f"  {'Trading Days':<36} {r['trading_days']:>10}",
        f"  {'Avg Positions / Day':<36} {r['avg_positions']:>10.1f}",
        f"  {'Avg Forward Return (selected)':<36} {afr_str}",
        "-" * 54,
        f"  Direction Accuracy (all non-flat):",
        f"    {fmt_acc(acc_all)}",
        f"  Direction Accuracy (selected top-{eq.shape[0] and 'N'} trades):",
        f"    {fmt_acc(acc_sel)}",
        "=" * 54,
        "",
    ]

    print("\n".join(lines))

    # Max drawdown from equity curve.
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    max_drawdown_pct = float(drawdown.min()) * 100

    summary = {
        "starting_capital": 1000.0,
        "final_value": r["final_value"],
        "total_return_pct": r["total_return_pct"],
        "max_drawdown_pct": max_drawdown_pct,
        "trading_days": r["trading_days"],
        "avg_positions": r["avg_positions"],
        "avg_forward_return_selected": r["avg_forward_return_selected"],
        "direction_accuracy_all": acc_all,
        "direction_accuracy_selected": acc_sel,
        "benchmark_return_pct": r["benchmark_return_pct"],
        "daily_equity": eq,
    }

    return summary
