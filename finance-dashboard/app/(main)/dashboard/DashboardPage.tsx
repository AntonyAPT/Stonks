"use client";

import { useState, useEffect } from "react";
import {
  Search,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  BarChart3,
} from "lucide-react";
import { useRouter } from "next/navigation";
import type { DashboardWatchlistItem } from "./page";
import type { StockQuote } from "@/app/api/stockquote/route";
import type { ModelRecommendation } from "@/types/model-recommendations";
import { PortfolioPanel } from "./PortfolioPanel";
import type { PortfolioItem } from "./PortfolioPanel";

export default function DashboardPage({
  watchlistItems,
  portfolioItems,
}: {
  watchlistItems: DashboardWatchlistItem[];
  portfolioItems: PortfolioItem[];
}) {
  const router = useRouter();
  const [search, setSearch] = useState("");
  const [allQuotes, setAllQuotes] = useState<Record<string, StockQuote>>({});
  const [recommendations, setRecommendations] = useState<ModelRecommendation[]>([]);
  const [recommendationsLoading, setRecommendationsLoading] = useState(true);
  const [visiblePredictionLimit, setVisiblePredictionLimit] = useState(4);

  // Compute net holdings (shares per ticker) from portfolio items
  const holdingsMap: Record<string, number> = {};
  for (const item of portfolioItems) {
    const delta = item.transaction_type === "sell" ? -item.quantity : item.quantity;
    holdingsMap[item.ticker] = (holdingsMap[item.ticker] ?? 0) + delta;
  }
  const activeHoldings = Object.entries(holdingsMap).filter(([, shares]) => shares > 0);

  useEffect(() => {
    const watchlistTickers = watchlistItems.map((i) => i.ticker).filter(Boolean);
    const portfolioTickers = Object.entries(
      portfolioItems.reduce((map, item) => {
        const delta = item.transaction_type === "sell" ? -item.quantity : item.quantity;
        map[item.ticker] = (map[item.ticker] ?? 0) + delta;
        return map;
      }, {} as Record<string, number>)
    )
      .filter(([, shares]) => shares > 0)
      .map(([ticker]) => ticker);

    const tickers = [...new Set([...watchlistTickers, ...portfolioTickers])];
    if (tickers.length === 0) return;

    fetch(`/api/stockquote?symbols=${tickers.join(",")}`)
      .then((r) => r.json())
      .then((data: StockQuote[]) => {
        const map: Record<string, StockQuote> = {};
        data.forEach((q) => { map[q.symbol] = q; });
        setAllQuotes(map);
      })
      .catch(console.error);
  }, [watchlistItems, portfolioItems]);

  useEffect(() => {
    fetch("/api/recommendations?forecast_day=5&direction=up&sort_by=prob_up&limit=10")
      .then((r) => r.json())
      .then((data: ModelRecommendation[] | { error?: string }) => {
        if (Array.isArray(data)) {
          setRecommendations(data);
        } else {
          console.error(data.error ?? "Failed to load recommendations");
          setRecommendations([]);
        }
      })
      .catch((err) => {
        console.error(err);
        setRecommendations([]);
      })
      .finally(() => setRecommendationsLoading(false));
  }, []);

  // Portfolio stats derived from live quotes
  const portfolioValue = activeHoldings.reduce(
    (sum, [ticker, shares]) => sum + shares * (allQuotes[ticker]?.currentPrice ?? 0),
    0
  );
  const todayPnl = activeHoldings.reduce(
    (sum, [ticker, shares]) => sum + shares * (allQuotes[ticker]?.change ?? 0),
    0
  );
  const prevPortfolioValue = portfolioValue - todayPnl;
  const portfolioPctChange = prevPortfolioValue > 0 ? (todayPnl / prevPortfolioValue) * 100 : 0;
  const visibleRecommendations = recommendations.slice(0, visiblePredictionLimit);
  const avgConfidence =
    visibleRecommendations.length > 0
      ? visibleRecommendations.reduce((sum, item) => sum + item.probUp, 0) / visibleRecommendations.length
      : 0;

  // Price map passed to PortfolioPanel so it doesn't need to re-fetch
  const portfolioPriceMap: Record<string, number> = {};
  activeHoldings.forEach(([ticker]) => {
    portfolioPriceMap[ticker] = allQuotes[ticker]?.currentPrice ?? 0;
  });

  const handleSearch = (e: React.FormEvent) => {
  e.preventDefault();
  const t = search.trim().toUpperCase();
  if (t) router.push(`/stocks/${t}`);
  };
  return (
    <div className="min-h-screen bg-page text-foreground">
      {/* Search Bar */}
      <div className="px-8 pt-6 pb-4">
        <div className="max-w-7xl mx-auto">
        <form onSubmit={handleSearch} className="max-w-2xl">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search stocks, predictions..."
              className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl pl-12 pr-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all"
            />
          </div>
        </form>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="px-8 pb-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
            <p className="text-slate-400">
              Monitor your predictions and portfolio performance
            </p>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <StatCard
              label="Portfolio Value"
              value={portfolioValue > 0 ? `$${portfolioValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—"}
              change={portfolioValue > 0 ? `${portfolioPctChange >= 0 ? "+" : ""}${portfolioPctChange.toFixed(2)}%` : "—"}
              isPositive={portfolioPctChange >= 0}
            />
            <StatCard
              label="Today's P&L"
              value={portfolioValue > 0 ? `${todayPnl >= 0 ? "+" : ""}$${Math.abs(todayPnl).toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—"}
              change={portfolioValue > 0 ? `${portfolioPctChange >= 0 ? "+" : ""}${portfolioPctChange.toFixed(2)}%` : "—"}
              isPositive={todayPnl >= 0}
            />
            <StatCard
              label="Active Predictions"
              value={recommendationsLoading ? "—" : String(visibleRecommendations.length)}
              change={recommendations[0]?.contextEnd ?? "—"}
              isPositive={true}
            />
            <StatCard
              label="Avg Confidence"
              value={avgConfidence > 0 ? `${Math.round(avgConfidence * 100)}%` : "—"}
              change="5-day"
              isPositive={true}
            />
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Portfolio Panel */}
            <PortfolioPanel portfolioItems={portfolioItems} quoteMap={portfolioPriceMap} />

            {/* AI Predictions Panel */}
            <div className="glass rounded-2xl p-6">
              <div className="flex items-center justify-between gap-3 mb-6">
                <h2 className="text-xl font-semibold">Top 5-Day Gainers Forecast</h2>
                <select
                  aria-label="Prediction count"
                  value={visiblePredictionLimit}
                  onChange={(event) => setVisiblePredictionLimit(Number(event.target.value))}
                  className="bg-slate-800/70 border border-slate-700/60 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                >
                  <option value={4}>Top 4</option>
                  <option value={10}>Top 10</option>
                </select>
              </div>
              <div className="space-y-4">
                {recommendationsLoading ? (
                  <p className="text-slate-500 text-sm text-center py-6">
                    Loading predictions...
                  </p>
                ) : recommendations.length === 0 ? (
                  <p className="text-slate-500 text-sm text-center py-6">
                    No model predictions found
                  </p>
                ) : (
                  visibleRecommendations.map((item) => (
                    <PredictionCard
                      key={`${item.ticker}-${item.forecastDay}`}
                      ticker={item.ticker}
                      prediction={item.recommendation}
                      confidence={Math.round(item.probUp * 100)}
                      detail="5-day probability"
                      isPositive={item.recommendation !== "SELL"}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Recent Activity */}
            <div className="lg:col-span-2 glass rounded-2xl p-6">
              <h2 className="text-xl font-semibold mb-6">Recent Activity</h2>
              <div className="space-y-4">
                {recommendations.slice(0, 2).map((item) => (
                  <ActivityItem
                    key={`activity-${item.ticker}-${item.forecastDay}`}
                    action={`5-day ${item.recommendation} prediction generated`}
                    ticker={item.ticker}
                    time={item.contextEnd}
                    type="prediction"
                  />
                ))}
                <ActivityItem
                  action="Portfolio updated"
                  ticker="TSLA"
                  time="1 hour ago"
                  type="update"
                />
                <ActivityItem
                  action="Alert triggered"
                  ticker="NVDA"
                  time="3 hours ago"
                  type="alert"
                />
                <ActivityItem
                  action="New prediction generated"
                  ticker="AMZN"
                  time="5 hours ago"
                  type="prediction"
                />
              </div>
            </div>

            {/* Watchlist */}
            <div className="glass rounded-2xl p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold">Watchlist</h2>
                <button
                  onClick={() => router.push("/watchlist")}
                  className="text-xs text-slate-400 hover:text-blue-400 transition-colors"
                >
                  View all
                </button>
              </div>
              <div className="space-y-3">
                {watchlistItems.length === 0 ? (
                  <p className="text-slate-500 text-sm text-center py-6">
                    No stocks in your watchlist yet
                  </p>
                ) : (
                  watchlistItems.map((item) => {
                    const quote = allQuotes[item.ticker];
                    const isPositive = (quote?.changePercent ?? 0) >= 0;
                    return (
                      <WatchlistItem
                        key={item.ticker}
                        ticker={item.ticker}
                        price={quote ? `$${quote.currentPrice.toFixed(2)}` : "—"}
                        change={
                          quote
                            ? `${isPositive ? "+" : ""}${quote.changePercent.toFixed(2)}%`
                            : "—"
                        }
                        isPositive={isPositive}
                        onClick={() => router.push(`/stocks/${item.ticker}`)}
                      />
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// Stat Card Component
function StatCard({
  label,
  value,
  change,
  isPositive,
}: {
  label: string;
  value: string;
  change: string;
  isPositive: boolean;
}) {
  return (
    <div className="glass rounded-2xl p-6 card-hover">
      <p className="text-slate-400 text-sm mb-2">{label}</p>
      <div className="flex items-end justify-between">
        <h3 className="text-2xl font-bold">{value}</h3>
        <div
          className={`flex items-center gap-1 text-sm font-medium ${
            isPositive ? "text-positive" : "text-negative"
          }`}
        >
          {isPositive ? (
            <TrendingUp className="w-4 h-4" />
          ) : (
            <TrendingDown className="w-4 h-4" />
          )}
          {change}
        </div>
      </div>
    </div>
  );
}

// Prediction Card Component
function PredictionCard({
  ticker,
  prediction,
  confidence,
  detail,
  isPositive,
}: {
  ticker: string;
  prediction: string;
  confidence: number;
  detail: string;
  isPositive: boolean;
}) {
  const getPredictionClass = (pred: string) => {
    if (pred === "BUY") return "text-positive bg-emerald-500/10";
    if (pred === "SELL") return "text-negative bg-red-500/10";
    return "text-neutral bg-amber-500/10";
  };

  return (
    <div className="bg-slate-800/30 border border-slate-700/30 rounded-xl p-4 hover:bg-slate-800/50 transition-all">
      <div className="flex items-center justify-between mb-3">
        <span className="font-semibold">{ticker}</span>
        <span
          className={`text-sm ${
            isPositive ? "text-positive" : "text-negative"
          }`}
        >
          {confidence}%
        </span>
      </div>
      <div className="flex items-center justify-between">
        <span
          className={`text-xs font-medium px-2 py-1 rounded ${getPredictionClass(
            prediction
          )}`}
        >
          {prediction}
        </span>
        <span className="text-xs text-slate-400">{detail}</span>
      </div>
    </div>
  );
}

// Activity Item Component
function ActivityItem({
  action,
  ticker,
  time,
  type,
}: {
  action: string;
  ticker: string;
  time: string;
  type: string;
}) {
  const getIcon = () => {
    if (type === "prediction")
      return <BarChart3 className="w-4 h-4 text-positive" />;
    if (type === "alert")
      return <AlertCircle className="w-4 h-4 text-neutral" />;
    return <TrendingUp className="w-4 h-4 text-blue-400" />;
  };

  return (
    <div className="flex items-center gap-4 p-3 bg-slate-800/30 rounded-lg hover:bg-slate-800/50 transition-all">
      <div className="w-8 h-8 bg-slate-700/50 rounded-lg flex items-center justify-center">
        {getIcon()}
      </div>
      <div className="flex-1">
        <p className="text-sm font-medium">{action}</p>
        <p className="text-xs text-slate-400">
          {ticker} • {time}
        </p>
      </div>
    </div>
  );
}

// Watchlist Item Component
function WatchlistItem({
  ticker,
  price,
  change,
  isPositive,
  onClick,
}: {
  ticker: string;
  price: string;
  change: string;
  isPositive: boolean;
  onClick?: () => void;
}) {
  return (
    <div onClick={onClick} className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg hover:bg-slate-800/50 transition-all cursor-pointer">
      <div>
        <p className="font-medium">{ticker}</p>
        <p className="text-sm text-slate-400">{price}</p>
      </div>
      <span
        className={`text-sm font-medium ${
          isPositive ? "text-positive" : "text-negative"
        }`}
      >
        {change}
      </span>
    </div>
  );
}
