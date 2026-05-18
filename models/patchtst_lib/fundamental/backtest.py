"""Quarterly paper-trading backtest for the PatchTST fundamental pipeline.

The strategy
------------
At each quarterly decision date:
1.  Take all stocks predicted "up" by the model.
2.  Select the top-``top_n`` by confidence.
3.  Allocate capital according to ``weighting`` across the selected names.
4.  Apply each stock's ``forward_return`` (entry close → exit close) to
    compute the quarter's P&L.
5.  Roll the updated portfolio value into the next quarter.

Lookahead-bias contract
-----------------------
- ``decision_date`` = ``datadate + publish_lag_days`` — the earliest date a
  trader could have known the fundamentals for that quarter.
- ``forward_return`` = (close_exit − close_entry) / close_entry, where
  ``close_entry`` is the first close on or after ``decision_date`` and
  ``close_exit`` is the first close on or after
  ``decision_date + forecast_lag_days``.
- Selection uses only ``predicted_direction`` and ``confidence`` — no prices
  from the forecast period leak into the selection step.

Usage
-----
    from patchtst_lib.fundamental.backtest import (
        build_prediction_df, run_backtest, summarize_results
    )

    pred_df = build_prediction_df(logits, labels, dataset, prices_df)
    results = run_backtest(pred_df, starting_capital=1000, top_n=10)
    summary = summarize_results(results)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


CLASS_NAMES = {0: "down", 1: "flat", 2: "up"}
UP_CLASS = 2


# ---------------------------------------------------------------------------
# 1. Build prediction DataFrame
# ---------------------------------------------------------------------------

def build_prediction_df(
    logits: np.ndarray,
    labels: np.ndarray,
    dataset,
    prices_df: pd.DataFrame,
    *,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    close_col: str = "Close",
) -> pd.DataFrame:
    """Convert raw model outputs into a flat per-window prediction DataFrame.

    Parameters
    ----------
    logits:
        Shape ``(N, 1, 3)``.  Raw logits from the model (before softmax).
    labels:
        Shape ``(N, 1)``.  Ground-truth class indices (0/1/2).
    dataset:
        A ``QuarterlyFundamentalDataset`` supporting ``.metadata(i)``.
    prices_df:
        OHLCV DataFrame with at least [date_col, ticker_col, close_col].
        Used *only* to compute ``forward_return``; no future prices leak into
        the selection step.

    Returns
    -------
    pd.DataFrame with columns:

    - ``decision_date``      — datadate + publish_lag (trade entry date)
    - ``forecast_end_date``  — decision_date + forecast_lag
    - ``ticker``
    - ``predicted_class``    — 0 / 1 / 2
    - ``predicted_direction``— "down" / "flat" / "up"
    - ``confidence``         — softmax probability for the predicted class
    - ``actual_class``       — 0 / 1 / 2
    - ``actual_direction``   — "down" / "flat" / "up"
    - ``forward_return``     — (close_exit − close_entry) / close_entry, NaN
                               when prices are missing
    """
    prices_df = prices_df.copy()
    prices_df[date_col] = pd.to_datetime(prices_df[date_col])
    price_lookup: dict = (
        prices_df.set_index([ticker_col, date_col])[close_col].to_dict()
    )

    # For asof lookups (first price >= target date).
    price_by_ticker: dict[str, pd.Series] = {}
    for ticker, grp in prices_df.groupby(ticker_col, sort=False):
        grp = grp.sort_values(date_col).set_index(date_col)
        price_by_ticker[str(ticker)] = grp[close_col].astype(float)

    def _asof_close(ticker: str, target_date) -> Optional[float]:
        series = price_by_ticker.get(ticker)
        if series is None:
            return None
        valid = series[series.index >= pd.Timestamp(target_date)]
        return float(valid.iloc[0]) if not valid.empty else None

    probs = _softmax(logits, axis=-1)  # (N, 1, 3)

    rows = []
    for i in range(len(dataset)):
        meta = dataset.metadata(i)

        pred_class = int(np.argmax(logits[i, 0]))
        actual_class = int(labels[i, 0])
        confidence = float(probs[i, 0, pred_class])

        entry_close = _asof_close(meta.ticker, meta.decision_date)
        exit_close  = _asof_close(meta.ticker, meta.forecast_end_date)

        if (
            entry_close is not None
            and exit_close is not None
            and entry_close > 0
        ):
            forward_return = (exit_close - entry_close) / entry_close
        else:
            forward_return = np.nan

        rows.append(
            {
                "decision_date": meta.decision_date,
                "forecast_end_date": meta.forecast_end_date,
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
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["forecast_end_date"] = pd.to_datetime(df["forecast_end_date"])
    return df.sort_values(["decision_date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Direction accuracy
# ---------------------------------------------------------------------------

def compute_direction_accuracy(df: pd.DataFrame) -> dict:
    """Up/down direction accuracy, excluding flat predictions and actuals."""
    mask = (df["predicted_direction"] != "flat") & (df["actual_direction"] != "flat")
    filtered = df[mask]
    total = len(filtered)
    if total == 0:
        return {"total_trades": 0, "correct": 0, "incorrect": 0, "accuracy": None}
    correct = int((filtered["predicted_direction"] == filtered["actual_direction"]).sum())
    return {
        "total_trades": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
    }


# ---------------------------------------------------------------------------
# 3. Top-N selection (reused from technical pipeline)
# ---------------------------------------------------------------------------

def select_top_confident_up(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top-N predicted-up stocks ranked by confidence (descending)."""
    up_preds = df[df["predicted_direction"] == "up"].copy()
    return up_preds.nlargest(top_n, "confidence", keep="first")


# ---------------------------------------------------------------------------
# 4. Position weighting
# ---------------------------------------------------------------------------

def _compute_weights(confidence: np.ndarray, weighting: str) -> np.ndarray:
    n = len(confidence)
    if n == 0:
        return np.array([])
    if weighting == "equal":
        return np.full(n, 1.0 / n)
    if weighting == "confidence":
        total = confidence.sum()
        return confidence / total if total > 0 else np.full(n, 1.0 / n)
    if weighting == "rank":
        ranks = np.arange(n, 0, -1, dtype=float)
        return ranks / ranks.sum()
    raise ValueError(f"Unknown weighting {weighting!r}. Choose 'equal', 'confidence', or 'rank'.")


# ---------------------------------------------------------------------------
# 5. Quarterly backtest
# ---------------------------------------------------------------------------

def _compute_quarterly_turnover(history: list[set]) -> float:
    """Average fraction of holdings that change quarter over quarter."""
    if len(history) < 2:
        return float("nan")
    turnovers = []
    for prev, curr in zip(history[:-1], history[1:]):
        union = prev | curr
        if not union:
            continue
        turnovers.append(len(prev.symmetric_difference(curr)) / len(union))
    return float(np.mean(turnovers)) if turnovers else float("nan")


def run_backtest(
    df: pd.DataFrame,
    starting_capital: float = 1000.0,
    top_n: int = 10,
    weighting: str = "equal",
    benchmark_prices: Optional[pd.Series] = None,
) -> dict:
    """Simulate a quarterly paper-trade strategy on the prediction DataFrame.

    Parameters
    ----------
    df:
        Output of ``build_prediction_df``.  Rows with NaN ``forward_return``
        are dropped before processing.
    starting_capital:
        Initial portfolio value.
    top_n:
        Number of top-confidence "up" stocks to hold each quarter.
    weighting:
        ``"equal"`` / ``"confidence"`` / ``"rank"``.
    benchmark_prices:
        Optional pd.Series indexed by date with a benchmark price (e.g. SPY
        quarterly closes).

    Returns
    -------
    dict with keys:
        - ``quarterly_equity``       — pd.Series(decision_date → portfolio_value)
        - ``final_value``
        - ``total_return_pct``
        - ``trading_quarters``
        - ``avg_positions``
        - ``avg_quarterly_return_selected``
        - ``annualized_sharpe``       — Sharpe × sqrt(4)
        - ``max_drawdown_pct``
        - ``avg_quarterly_turnover``
        - ``direction_accuracy_all``
        - ``direction_accuracy_selected``
        - ``benchmark_return_pct``   — None if benchmark not supplied
        - ``weighting``
    """
    df = df.dropna(subset=["forward_return"]).copy()
    if df.empty:
        raise ValueError("No rows with valid forward_return — cannot run backtest.")

    portfolio_value = starting_capital
    quarterly_records: list[dict] = []
    positions_per_quarter: list[int] = []
    selected_frames: list[pd.DataFrame] = []
    holdings_history: list[set] = []

    for decision_date in sorted(df["decision_date"].unique()):
        quarter_preds = df[df["decision_date"] == decision_date]
        selected = select_top_confident_up(quarter_preds, top_n)
        n_positions = len(selected)
        positions_per_quarter.append(n_positions)
        holdings_history.append(set(selected["ticker"].tolist()))

        if n_positions > 0:
            weights = _compute_weights(selected["confidence"].values, weighting)
            quarter_return = float((selected["forward_return"].values * weights).sum())
            portfolio_value *= 1.0 + quarter_return
            selected_frames.append(selected)

        quarterly_records.append(
            {
                "date": decision_date,
                "portfolio_value": portfolio_value,
                "n_positions": n_positions,
            }
        )

    quarterly_df = pd.DataFrame(quarterly_records)
    quarterly_equity = quarterly_df.set_index("date")["portfolio_value"]

    acc_all = compute_direction_accuracy(df)

    if selected_frames:
        traded = pd.concat(selected_frames, ignore_index=True)
        acc_selected = compute_direction_accuracy(traded)
        quarterly_returns = pd.concat(
            [s.assign(_w=_compute_weights(s["confidence"].values, weighting))
               for s in selected_frames],
            ignore_index=True,
        )
        per_quarter_returns = [
            (s["forward_return"].values * _compute_weights(s["confidence"].values, weighting)).sum()
            for s in selected_frames
        ]
        avg_qret = float(np.mean(per_quarter_returns))
        std_qret = float(np.std(per_quarter_returns, ddof=1)) if len(per_quarter_returns) > 1 else float("nan")
        annualized_sharpe = (avg_qret / std_qret * np.sqrt(4)) if std_qret > 0 else float("nan")
    else:
        acc_selected = {"total_trades": 0, "correct": 0, "incorrect": 0, "accuracy": None}
        avg_qret = float("nan")
        annualized_sharpe = float("nan")

    # Max drawdown.
    running_max = quarterly_equity.cummax()
    drawdown = (quarterly_equity - running_max) / running_max
    max_drawdown_pct = float(drawdown.min()) * 100

    # Benchmark.
    benchmark_return_pct = None
    if benchmark_prices is not None:
        bench = benchmark_prices.sort_index()
        start_date = quarterly_equity.index.min()
        end_date = quarterly_equity.index.max()
        bench_window = bench.loc[(bench.index >= start_date) & (bench.index <= end_date)]
        if len(bench_window) >= 2:
            benchmark_return_pct = (bench_window.iloc[-1] / bench_window.iloc[0] - 1) * 100

    return {
        "quarterly_equity": quarterly_equity,
        "final_value": portfolio_value,
        "total_return_pct": (portfolio_value / starting_capital - 1) * 100,
        "trading_quarters": len(quarterly_records),
        "avg_positions": float(np.mean(positions_per_quarter)) if positions_per_quarter else 0.0,
        "avg_quarterly_return_selected": avg_qret,
        "annualized_sharpe": annualized_sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_quarterly_turnover": _compute_quarterly_turnover(holdings_history),
        "direction_accuracy_all": acc_all,
        "direction_accuracy_selected": acc_selected,
        "benchmark_return_pct": benchmark_return_pct,
        "weighting": weighting,
    }


# ---------------------------------------------------------------------------
# 6. Results summary
# ---------------------------------------------------------------------------

def summarize_results(backtest_results: dict) -> dict:
    """Print a formatted quarterly backtest summary and return it as a dict."""
    r = backtest_results
    eq = r["quarterly_equity"]

    def fmt_acc(acc_dict: dict) -> str:
        if acc_dict["accuracy"] is None:
            return "N/A"
        return (
            f"{acc_dict['accuracy']:.2%}  "
            f"({acc_dict['correct']}/{acc_dict['total_trades']} correct)"
        )

    sharpe = r["annualized_sharpe"]
    sharpe_str = f"{sharpe:>+8.3f}" if not np.isnan(sharpe) else "     N/A"
    turnover = r["avg_quarterly_turnover"]
    turnover_str = f"{turnover:.1%}" if not np.isnan(turnover) else "N/A"
    avg_qret = r["avg_quarterly_return_selected"]
    avg_qret_str = f"{avg_qret:>+10.4%}" if not np.isnan(avg_qret) else "       N/A"

    lines = [
        "",
        "=" * 60,
        "  QUARTERLY FUNDAMENTAL BACKTEST RESULTS",
        "=" * 60,
        f"  {'Weighting scheme':<40} {r['weighting']:>10}",
        f"  {'Starting Capital':<40} ${1000.0:>10,.2f}",
        f"  {'Final Portfolio Value':<40} ${r['final_value']:>10,.2f}",
        f"  {'Total Return':<40} {r['total_return_pct']:>+10.2f}%",
    ]

    if r["benchmark_return_pct"] is not None:
        lines.append(
            f"  {'Benchmark Return (buy & hold)':<40} "
            f"{r['benchmark_return_pct']:>+10.2f}%"
        )

    lines += [
        f"  {'Trading Quarters':<40} {r['trading_quarters']:>10}",
        f"  {'Avg Positions / Quarter':<40} {r['avg_positions']:>10.1f}",
        f"  {'Avg Quarterly Return (selected)':<40} {avg_qret_str}",
        f"  {'Annualised Sharpe (×√4)':<40} {sharpe_str}",
        f"  {'Max Drawdown':<40} {r['max_drawdown_pct']:>+10.2f}%",
        f"  {'Avg Quarterly Turnover':<40} {turnover_str:>10}",
        "-" * 60,
        f"  Direction Accuracy (all non-flat):",
        f"    {fmt_acc(r['direction_accuracy_all'])}",
        f"  Direction Accuracy (selected top-N trades):",
        f"    {fmt_acc(r['direction_accuracy_selected'])}",
        "=" * 60,
        "",
    ]

    print("\n".join(lines))

    return {
        "weighting": r["weighting"],
        "starting_capital": 1000.0,
        "final_value": r["final_value"],
        "total_return_pct": r["total_return_pct"],
        "max_drawdown_pct": r["max_drawdown_pct"],
        "trading_quarters": r["trading_quarters"],
        "avg_positions": r["avg_positions"],
        "avg_quarterly_return_selected": r["avg_quarterly_return_selected"],
        "annualized_sharpe": r["annualized_sharpe"],
        "avg_quarterly_turnover": r["avg_quarterly_turnover"],
        "direction_accuracy_all": r["direction_accuracy_all"],
        "direction_accuracy_selected": r["direction_accuracy_selected"],
        "benchmark_return_pct": r["benchmark_return_pct"],
        "quarterly_equity": eq,
    }
