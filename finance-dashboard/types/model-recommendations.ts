export interface ModelRecommendation {
  ticker: string;
  sector: string | null;
  industry: string | null;
  contextEnd: string;
  forecastDay: number;
  forecastDate: string;
  predictedClass: number;
  predictedDirection: "down" | "flat" | "up";
  recommendation: "SELL" | "HOLD" | "BUY";
  confidence: number;
  probDown: number;
  probFlat: number;
  probUp: number;
  lastClose: number | null;
  runTimestamp: string;
}

export interface DbRecommendationRow {
  ticker: string;
  sector: string | null;
  industry: string | null;
  context_end: string;
  forecast_day: number;
  forecast_date: string;
  predicted_class: number;
  predicted_direction: "down" | "flat" | "up";
  recommendation: "SELL" | "HOLD" | "BUY";
  confidence: number;
  prob_down: number;
  prob_flat: number;
  prob_up: number;
  last_close: number | null;
  run_timestamp: string;
}
