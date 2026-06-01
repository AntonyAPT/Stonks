"""Run weighting-strategy comparisons locally using a saved pred_df CSV.

No model, no GPU, no Kaggle needed.  Pull pred_df_test.csv first:
    bash pull_results.sh

Then run:
    python backtest_local.py
    python backtest_local.py --top_n 5
    python backtest_local.py --weighting confidence rank equal
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import backtest as bt

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent / "save_dir" / "pred_df_test.csv"),
        help="Path to pred_df_test.csv downloaded from Kaggle.",
    )
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Minimum softmax confidence to enter a position (default: 0.5).")
    parser.add_argument(
        "--weighting",
        nargs="+",
        default=["equal", "confidence", "rank"],
        choices=["equal", "confidence", "rank"],
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Run `bash pull_results.sh` from models/notebook first.")
        raise SystemExit(1)

    pred_df = pd.read_csv(csv_path, parse_dates=["trade_date", "forecast_date"])
    print(f"Loaded {len(pred_df):,} prediction rows from {csv_path.name}")
    print(f"Date range: {pred_df['trade_date'].min().date()} -> {pred_df['trade_date'].max().date()}\n")

    colors = {"equal": "tab:blue", "confidence": "tab:orange", "rank": "tab:green"}
    fig, ax = plt.subplots(figsize=(13, 5))
    summaries = {}

    for w in args.weighting:
        res = bt.run_backtest(pred_df, starting_capital=args.capital, top_n=args.top_n, weighting=w)
        summaries[w] = bt.summarize_results(res)
        eq = res["daily_equity"]
        ret = res["total_return_pct"]
        eq.plot(ax=ax, color=colors.get(w, None), linewidth=1.5, label=f"{w}  ({ret:+.1f}%)")

    ax.axhline(args.capital, color="gray", linestyle="--", linewidth=0.8, label=f"${args.capital:,.0f} baseline")
    ax.set_title(f"Equity Curve by Weighting — Top-{args.top_n} Confidence Up Picks")
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nReturn summary:")
    for w, s in summaries.items():
        print(f"  {w:<12} {s['total_return_pct']:>+7.2f}%   "
              f"max_dd {s['max_drawdown_pct']:>+6.2f}%   "
              f"avg_fwd_ret {s['avg_forward_return_selected']:>+.4%}")


if __name__ == "__main__":
    main()
