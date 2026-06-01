import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";
import type { DbRecommendationRow, ModelRecommendation } from "@/types/model-recommendations";

export async function GET(req: NextRequest) {
  const supabase = await createClient();
  const forecastDay = Number(req.nextUrl.searchParams.get("forecast_day") ?? "1");
  const limit = Math.min(Number(req.nextUrl.searchParams.get("limit") ?? "8"), 50);
  const direction = req.nextUrl.searchParams.get("direction");
  const sortBy = req.nextUrl.searchParams.get("sort_by") ?? "confidence";
  const allowedDirections = new Set(["down", "flat", "up"]);
  const allowedSortColumns = new Set(["confidence", "prob_down", "prob_flat", "prob_up"]);

  if (!Number.isInteger(forecastDay) || forecastDay < 1 || forecastDay > 5) {
    return NextResponse.json({ error: "forecast_day must be 1-5" }, { status: 400 });
  }

  if (direction && !allowedDirections.has(direction)) {
    return NextResponse.json({ error: "direction must be down, flat, or up" }, { status: 400 });
  }

  if (!allowedSortColumns.has(sortBy)) {
    return NextResponse.json({ error: "sort_by must be confidence, prob_down, prob_flat, or prob_up" }, { status: 400 });
  }

  const { data: latest, error: latestError } = await supabase
    .from("model_recommendations")
    .select("context_end")
    .order("context_end", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (latestError) {
    console.error("[/api/recommendations] latest context error:", latestError.message);
    return NextResponse.json({ error: latestError.message }, { status: 500 });
  }

  if (!latest?.context_end) {
    return NextResponse.json([]);
  }

  let query = supabase
    .from("model_recommendations")
    .select(
      "ticker, sector, industry, context_end, forecast_day, forecast_date, predicted_class, predicted_direction, recommendation, confidence, prob_down, prob_flat, prob_up, last_close, run_timestamp"
    )
    .eq("context_end", latest.context_end)
    .eq("forecast_day", forecastDay);

  if (direction) {
    query = query.eq("predicted_direction", direction);
  }

  const { data, error } = await query
    .order(sortBy, { ascending: false })
    .order("confidence", { ascending: false })
    .limit(limit);

  if (error) {
    console.error("[/api/recommendations] query error:", error.message);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  const rows: ModelRecommendation[] = ((data ?? []) as DbRecommendationRow[]).map((row) => ({
    ticker: row.ticker,
    sector: row.sector,
    industry: row.industry,
    contextEnd: row.context_end,
    forecastDay: row.forecast_day,
    forecastDate: row.forecast_date,
    predictedClass: row.predicted_class,
    predictedDirection: row.predicted_direction,
    recommendation: row.recommendation,
    confidence: row.confidence,
    probDown: row.prob_down,
    probFlat: row.prob_flat,
    probUp: row.prob_up,
    lastClose: row.last_close,
    runTimestamp: row.run_timestamp,
  }));

  return NextResponse.json(rows);
}
