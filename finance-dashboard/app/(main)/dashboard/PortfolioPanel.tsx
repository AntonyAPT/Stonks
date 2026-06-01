"use client";

import { useState, useEffect } from "react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import type { StockQuote } from "@/app/api/stockquote/route";

export type PortfolioItem = {
  ticker: string;
  quantity: number;
  buy_price: number;
  buy_date: string;
  transaction_type: string;
};

const COLORS = [
  "#38bdf8",
  "#818cf8",
  "#34d399",
  "#fb923c",
  "#f472b6",
  "#a78bfa",
  "#facc15",
  "#4ade80",
];

function buildGrowthData(items: PortfolioItem[]) {
  let running = 0;
  const points: { date: string; value: number }[] = [];

  for (const item of items) {
    const delta =
      item.transaction_type === "sell"
        ? -(item.quantity * item.buy_price)
        : item.quantity * item.buy_price;
    running += delta;
    const date = new Date(item.buy_date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });
    if (points.length > 0 && points[points.length - 1].date === date) {
      points[points.length - 1].value = parseFloat(running.toFixed(2));
    } else {
      points.push({ date, value: parseFloat(running.toFixed(2)) });
    }
  }
  return points;
}

export function PortfolioPanel({
  portfolioItems,
  quoteMap: externalQuoteMap,
}: {
  portfolioItems: PortfolioItem[];
  quoteMap?: Record<string, number>;
}) {
  const [activeTab, setActiveTab] = useState<"composition" | "overview">(
    "composition"
  );
  const [internalQuotes, setInternalQuotes] = useState<Record<string, number>>({});

  // Compute current holdings (net shares > 0)
  const holdingsMap: Record<string, number> = {};
  for (const item of portfolioItems) {
    const delta =
      item.transaction_type === "sell" ? -item.quantity : item.quantity;
    holdingsMap[item.ticker] = (holdingsMap[item.ticker] ?? 0) + delta;
  }
  const holdings = Object.entries(holdingsMap)
    .filter(([, shares]) => shares > 0)
    .map(([ticker, shares]) => ({ ticker, shares }));

  useEffect(() => {
    if (externalQuoteMap) return; // parent already fetched quotes
    if (holdings.length === 0) return;
    const tickers = holdings.map((h) => h.ticker).join(",");
    fetch(`/api/stockquote?symbols=${tickers}`)
      .then((r) => r.json())
      .then((data: StockQuote[]) => {
        const map: Record<string, number> = {};
        data.forEach((q) => {
          map[q.symbol] = q.currentPrice;
        });
        setInternalQuotes(map);
      })
      .catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portfolioItems, externalQuoteMap]);

  const quotes = externalQuoteMap ?? internalQuotes;

  // Pie data: current price * shares, fallback to avg buy price
  const pieData = holdings
    .map(({ ticker, shares }) => {
      const buys = portfolioItems.filter(
        (i) => i.ticker === ticker && i.transaction_type === "buy"
      );
      const avgBuy =
        buys.length > 0
          ? buys.reduce((s, i) => s + i.buy_price, 0) / buys.length
          : 0;
      const price = quotes[ticker] ?? avgBuy;
      return { name: ticker, value: parseFloat((shares * price).toFixed(2)) };
    })
    .filter((d) => d.value > 0);

  const totalValue = pieData.reduce((s, d) => s + d.value, 0);
  const growthData = buildGrowthData(portfolioItems);
  const isEmpty = portfolioItems.length === 0;

  return (
    <div className="lg:col-span-2 glass rounded-2xl p-6">
      {/* Tab Header */}
      <div className="flex items-center gap-1 mb-6">
        <button
          onClick={() => setActiveTab("composition")}
          className={`px-4 py-2 text-sm rounded-lg transition-colors ${
            activeTab === "composition"
              ? "bg-blue-500/20 text-blue-400"
              : "text-slate-400 hover:bg-slate-800/50"
          }`}
        >
          Composition
        </button>
        <button
          onClick={() => setActiveTab("overview")}
          className={`px-4 py-2 text-sm rounded-lg transition-colors ${
            activeTab === "overview"
              ? "bg-blue-500/20 text-blue-400"
              : "text-slate-400 hover:bg-slate-800/50"
          }`}
        >
          Overview
        </button>
      </div>

      {isEmpty ? (
        <div className="h-80 flex items-center justify-center text-slate-500 text-sm">
          No holdings in your portfolio yet
        </div>
      ) : activeTab === "composition" ? (
        <CompositionTab pieData={pieData} totalValue={totalValue} />
      ) : (
        <OverviewTab growthData={growthData} />
      )}
    </div>
  );
}

function CompositionTab({
  pieData,
  totalValue,
}: {
  pieData: { name: string; value: number }[];
  totalValue: number;
}) {
  if (pieData.length === 0) {
    return (
      <div className="h-100 flex items-center justify-center text-slate-500 text-sm">
        No holdings to display
      </div>
    );
  }

  return (
    <div className="h-100 flex items-center gap-6">
      <div className="w-[55%] h-full pointer-events-none">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius="52%"
              outerRadius="78%"
              dataKey="value"
              paddingAngle={2}
              isAnimationActive={false}
            >
              {pieData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex-1 space-y-2.5 overflow-y-auto max-h-72 pr-1">
        <p className="text-xs text-slate-400 mb-3">
          All Portfolios:{" "}
          <span className="text-white font-semibold">
            $
            {totalValue.toLocaleString(undefined, {
              maximumFractionDigits: 0,
            })}
          </span>
        </p>
        {pieData.map((entry, i) => {
          const pct = ((entry.value / totalValue) * 100).toFixed(1);
          return (
            <div key={entry.name} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span
                  className="w-2.5 h-2.5 rounded-full shrink-0"
                  style={{ backgroundColor: COLORS[i % COLORS.length] }}
                />
                <span className="text-sm font-medium">{entry.name}</span>
              </div>
              <span className="text-sm text-slate-400">{pct}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function OverviewTab({
  growthData,
}: {
  growthData: { date: string; value: number }[];
}) {
  if (growthData.length === 0) {
    return (
      <div className="h-80 flex items-center justify-center text-slate-500 text-sm">
        No transactions yet
      </div>
    );
  }

  const isPositive =
    growthData.length < 2 ||
    growthData[growthData.length - 1].value >= growthData[0].value;

  const color = isPositive ? "#34d399" : "#f87171";

  return (
    <div className="h-100 flex items-center justify-center">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={growthData}
          margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
        >
          <defs>
            <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(148, 163, 184, 0.16)" vertical={false} />
          <XAxis
            dataKey="date"
            stroke="#94a3b8"
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            stroke="#94a3b8"
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) =>
              v >= 1000 ? `$${(v / 1000).toFixed(0)}k` : `$${v}`
            }
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#020617",
              border: "1px solid rgba(51, 65, 85, 0.9)",
              borderRadius: "12px",
            }}
            formatter={(value: number | undefined) => [
              `$${(value ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}`,
              "Portfolio Value",
            ]}
            labelStyle={{ color: "#e2e8f0" }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            fill="url(#portfolioGradient)"
            dot={false}
            activeDot={{ r: 4, fill: color }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
