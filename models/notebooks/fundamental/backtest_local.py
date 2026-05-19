"""Run quarterly weighting-strategy comparisons locally using a saved pred_df CSV.

No model, no GPU, no Kaggle needed.  Pull pred_df_test.csv first:
    bash pull_results.sh

Then run:
    python backtest_local.py
    python backtest_local.py --top_n 5
    python backtest_local.py --weighting confidence rank equal
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Add models/ to sys.path so patchtst_lib is importable when running this
# script directly from the models/notebooks/fundamental/ directory.
sys.path.insert(0, str(Path(__file__).parents[2]))

import patchtst_lib.fundamental.backtest as fbt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent / "save_dir_fund" / "pred_df_test.csv"),
        help="Path to pred_df_test.csv downloaded from Kaggle.",
    )
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument(
        "--weighting",
        nargs="+",
        default=["equal", "confidence", "rank"],
        choices=["equal", "confidence", "rank"],
    )
    parser.add_argument(
        "--forecast-year",
        type=int,
        default=None,
        help=(
            "Calendar year of Q1 to evaluate (e.g. 2025 = Q4-2024 fundamentals "
            "predicting Q1-2025). Default: latest year in the CSV."
        ),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=12,
        help="Model context length in quarters (must match training).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Run `bash pull_results.sh` from models/notebooks/fundamental first.")
        raise SystemExit(1)

    pred_df = pd.read_csv(
        csv_path,
        parse_dates=["decision_date", "forecast_end_date"],
    )
    print(f"Loaded {len(pred_df):,} prediction rows from {csv_path.name}")
    print(
        f"Decision-date range: {pred_df['decision_date'].min().date()} "
        f"-> {pred_df['decision_date'].max().date()}"
    )

    pred_df_bt = fbt.filter_prior_year_to_following_q1(
        pred_df, forecast_year=args.forecast_year
    )
    if pred_df_bt.empty:
        print("ERROR: No Q4→following-Q1 rows after filtering.")
        raise SystemExit(1)

    fy = int(pred_df_bt["forecast_year"].iloc[0])
    cy = int(pred_df_bt["context_year"].iloc[0])
    ctx = args.context_length
    print(
        f"Backtest: 1 rebalance for Q1-{fy} "
        f"(model input = {ctx} quarters ending Q4-{cy}; "
        f"{len(pred_df_bt):,} ticker-level predictions)\n"
    )

    bar_colors = {"equal": "tab:blue", "confidence": "tab:orange", "rank": "tab:green"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    summaries = {}

    for w in args.weighting:
        res = fbt.run_backtest(
            pred_df_bt,
            starting_capital=args.capital,
            top_n=args.top_n,
            weighting=w,
        )
        summaries[w] = fbt.summarize_results(res)

    labels = list(summaries.keys())
    returns = [summaries[w]["total_return_pct"] for w in labels]
    ax.bar(labels, returns, color=[bar_colors[w] for w in labels])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Total Return (%)")
    ax.set_title(
        f"Single-period backtest: Q1 {fy} return after Q4 {cy} earnings\n"
        f"{ctx}-quarter model context (ends Q4 {cy}) · 1 rebalance · "
        f"top-{args.top_n} predicted-up stocks",
        fontsize=11,
    )
    # fig.text(
    #     0.5,
    #     -0.06,
    #     "Holdings from Q4-anchored windows; return = weighted forward_return "
    #     "over ~63 days after 45-day filing lag.",
    #     ha="center",
    #     fontsize=9,
    #     color="gray",
    # )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nReturn summary:")
    for w, s in summaries.items():
        sharpe = s["annualized_sharpe"]
        sharpe_str = f"{sharpe:>+7.3f}" if sharpe is not None and pd.notna(sharpe) else "    N/A"
        print(
            f"  {w:<12} {s['total_return_pct']:>+7.2f}%   "
            f"max_dd {s['max_drawdown_pct']:>+6.2f}%   "
            f"avg_q_ret {s['avg_quarterly_return_selected']:>+.4%}   "
            f"sharpe {sharpe_str}"
        )


if __name__ == "__main__":
    main()
