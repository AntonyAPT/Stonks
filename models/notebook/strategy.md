b) How to change the strategy
Number of stocks — just change top_n:


results = bt.run_backtest(pred_df_1y, top_n=5)   # hold 5 stocks
results = bt.run_backtest(pred_df_1y, top_n=20)  # hold 20 stocks
Weighting by confidence (most confident gets more money) — the current code does not support this yet. It uses pure equal weight. To support confidence-weighted sizing, run_backtest needs one new parameter. I can add this now — it would look like:


results = bt.run_backtest(pred_df_1y, top_n=10, weighting="equal")       # current
results = bt.run_backtest(pred_df_1y, top_n=10, weighting="confidence")  # rank-weighted
results = bt.run_backtest(pred_df_1y, top_n=10, weighting="softmax")     # softmax-scaled

c) Current strategy — exactly what it does
Decision time: End of day trade_date (= context_end, the last day of the 128-day context window). The model has seen data up to and including this date's close. No future data is visible.

Buy: Effectively at the close of trade_date (or equivalently open of the next day). The top 10 stocks with predicted_direction == "up", selected by highest confidence score.

Sell: At the close of forecast_start — which is exactly 1 trading day later (the very next session after trade_date). This is a pure overnight / 1-day hold.

How much per stock: Exactly equal weight. If you hold 10 stocks and the portfolio is $1,000, each stock gets $100. If only 6 qualify that day, each gets $1,000 / 6 = $166.67. If 0 qualify, 100% cash, 0% return that day.

Return formula per stock:


(close[forecast_start] - close[trade_date]) / close[trade_date]
Portfolio update:


new_value = old_value × (1 + average of the 10 stocks' day returns)
What gets ignored: Any stock the model predicts "flat" or "down" is never bought. Confidence score is used only for ranking (picking top 10), not for sizing (all 10 get equal dollars).

Summary table:

Question	Answer
Hold period	1 trading day
Entry	Close of trade_date
Exit	Close of forecast_start (next day)
# of stocks	10 (configurable via top_n)
Sizing	Equal weight (1/N of portfolio each)
Selection filter	Must be predicted "up"
Tie-breaking	Highest confidence score wins
Cash days	When 0 stocks predicted "up"
The biggest lever you don't have yet is confidence-weighted sizing — the #1 stock gets proportionally more than #10. Want me to add that?