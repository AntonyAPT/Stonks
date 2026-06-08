# STONKS - Internal Documentation

This document is the onboarding reference for the STONKS senior project. It explains how the code is organized and how the pieces work together, so that a new contributor can get oriented quickly.

It is split into two main parts:

1. Project Structure - an annotated map of every meaningful file.
2. File Interactions - how those files work together and what features result.

A few supporting sections (overview, end-to-end data flow, environment variables, quick reference, and known gaps) have also been added. 

---

## 0. Overview and Tech Stack

STONKS is a minimalist finance dashboard for individual investors. Users sign in with Google, build and track multiple stock portfolios, maintain watchlists, view interactive price and fundamentals charts, and see daily machine-learning stock recommendations.

The project has three runtime pieces that all share a single Supabase (PostgreSQL + Auth) backend:

- Finance Dashboard - a Next.js 16 (App Router) web app the user interacts with. It reads and writes user data in Supabase and pulls live prices and search results from external market data APIs (Finnhub, Yahoo Finance, and embedded TradingView widgets).
- Model Training - a PatchTST time-series transformer pipeline (Python) that trains on Kaggle GPUs, runs local/automated inference, and publishes BUY / HOLD / SELL recommendations into Supabase.
- Automation - a GitHub Actions workflow that, on a schedule, refreshes market data and republishes the technical recommendations so the dashboard always shows fresh predictions.

Tech stack at a glance:

- Frontend: Next.js 16, React 18, TypeScript, Tailwind CSS v4, CSS Modules,
Recharts, TradingView widgets, Sonner (toasts).
- Backend / data: Supabase (PostgreSQL, Auth, Row Level Security).
- ML: PyTorch, HuggingFace Transformers (PatchTST), pandas, scikit-learn.
- Data sources: Finnhub (search, quotes), Yahoo Finance (history), yfinance
(training data ingestion).
- CI / automation: GitHub Actions; Kaggle kernels for GPU training.

---

## 1. Project Structure

Every meaningful source file gets a one-sentence description. Boilerplate (CSS Modules, barrel `index.ts` files, generated type files, and standard config files) is collapsed into grouped notes to keep the map readable. Build artifacts (`node_modules/`, `.next/`,`__pycache__/`, `.venv/`) and large data/model binaries are omitted or noted at the directory level only.

### 1a. Finance Dashboard (`finance-dashboard/`)

```
finance-dashboard/
|
+-- proxy.ts                         Next.js 16 request proxy (auth gate) that delegates session refresh and route protection.
+-- next.config.ts                   Next.js config allowing Google OAuth avatar images.
+-- package.json                     Declares dependencies and developer scripts.
+-- (other root config)             tsconfig.json, eslint.config.mjs, postcss.config.mjs, next-env.d.ts - standard tooling config.
|
+-- app/
|   +-- layout.tsx                   Root HTML shell with fonts, global CSS, and ThemeProvider.
|   +-- page.tsx                     Placeholder landing page at "/" (authenticated users are redirected to /dashboard).
|   +-- globals.css                  Tailwind theme tokens, dark/light variables, and shared UI utilities.
|   +-- contexts/
|   |   +-- ThemeContext.tsx         Client context persisting dark/light theme in localStorage.
|   |
|   +-- (auth)/                      Route group for public authentication pages.
|   |   +-- layout.tsx               Centered auth layout with card wrapper.
|   |   +-- sign-in/
|   |   |   +-- page.tsx             Sign-in landing page with logo and Google OAuth button.
|   |   |   +-- SignInButton.tsx     Client button that starts Supabase Google OAuth.
|   |   +-- auth/callback/route.ts   OAuth callback that exchanges the code, creates a profile row if missing, and redirects to /dashboard.
|   |   +-- components/              Logo.tsx plus a barrel index.ts and auth.module.css.
|   |
|   +-- (main)/                      Route group for authenticated app pages.
|   |   +-- layout.tsx               Auth-gated layout: verifies user, provides SelectedPortfolioProvider, navbar, footer, and toasts.
|   |   +-- actions/
|   |   |   +-- portfolio.ts         Server actions for portfolio CRUD, buy/sell trades, and historical performance.
|   |   +-- contexts/
|   |   |   +-- SelectedPortfolioContext.tsx  Client context remembering the user's selected portfolio across navigation.
|   |   +-- hooks/                   useNavigation.ts (route helper) plus a barrel index.ts.
|   |   +-- components/
|   |   |   +-- AddStockButton.tsx   Button that opens the shared stock search modal.
|   |   |   +-- navbar/              Navbar.tsx (server), NavLinks.tsx, UserMenu.tsx, plus navbar.module.css.
|   |   |
|   |   +-- dashboard/
|   |   |   +-- page.tsx             Server page fetching watchlist items and portfolio transactions.
|   |   |   +-- DashboardPage.tsx    Client dashboard: stats, AI prediction panels, watchlist preview, portfolio panel.
|   |   |   +-- PortfolioPanel.tsx   Client charts for aggregate composition and cost-basis growth.
|   |   |
|   |   +-- portfolios/
|   |   |   +-- page.tsx             Server page loading all portfolios with items and computed totals.
|   |   |   +-- PortfoliosPage.tsx   Client UI to list, select, create, rename, and delete portfolios.
|   |   |   +-- types.ts             Shared types for the portfolios list and selection setter.
|   |   |   +-- hooks/usePortfolioActions.ts  Client hook orchestrating create/rename/delete actions.
|   |   |   +-- components/          PortfolioItem.tsx, CreatePortfolioModal.tsx, DeleteConfirmModal.tsx, plus portfolios.module.css.
|   |   |
|   |   +-- portfolio/
|   |   |   +-- [id]/page.tsx        Server page for one portfolio: holdings, FIFO cost basis, live quotes, insights, ledger.
|   |   |   +-- components/          Chart and table components for a single portfolio (see grouped note below).
|   |   |
|   |   +-- watchlist/
|   |   |   +-- page.tsx             Server page loading default watchlist items enriched with company names.
|   |   |   +-- WatchlistPage.tsx    Client watchlist table with live quotes, refresh, add/remove, and summary stats.
|   |   |   +-- actions.ts           Server actions to add/remove watchlist items and ensure a default watchlist.
|   |   |   +-- components/AddStockModal.tsx  Search modal wired to addToWatchlist.
|   |   |
|   |   +-- stocks/
|   |       +-- page.tsx             Market overview page: stock search, TradingView heatmap, market widgets, top stories.
|   |       +-- [ticker]/page.tsx    Server stock detail page: quarterly fundamentals, TradingView widgets, buy/watchlist UI.
|   |
|   +-- api/                         Route Handlers (server-side JSON endpoints).
|       +-- stocksearch/route.ts     GET proxy to Finnhub symbol search, enriching top results with profile data.
|       +-- stockquote/route.ts      GET proxy fetching parallel Finnhub real-time quotes for given tickers.
|       +-- recommendations/route.ts            GET latest technical predictions from the model_recommendations table.
|       +-- fundamental-recommendations/route.ts GET latest fundamental predictions from fundamental_recommendations.
|       +-- quarterly/route.ts       GET wrapper around getQuarterlyFundamentals for a single ticker.
|
+-- components/                      Cross-feature reusable UI.
|   +-- stock-search/                StockSearch.tsx + useStockSearch.ts (debounced /api/stocksearch hook) + barrel index.ts.
|   +-- stocks/
|   |   +-- BuyAndWatchlist.tsx      Stock detail panel to buy into the selected portfolio or toggle watchlist membership.
|   |   +-- QuarterlyDataPanel.tsx   Year/quarter picker displaying formatted fundamental metrics.
|   |   +-- charts/                  Recharts panels for revenue, net income, EPS, and balance sheet (see grouped note below).
|   +-- tradingview/
|       +-- TradingViewWidget.tsx    SSR-safe wrapper around the TradingView widget library.
|       +-- StockDetailWidgets.tsx   Composes profile, symbol overview, and fundamentals widgets for a ticker.
|       +-- widget-configs.ts        Preset widget configurations for market and per-symbol views (plus barrel index.ts).
|
+-- lib/
|   +-- quarterly.ts                 Server-only helper querying quarterly_fundamentals by ticker (service role).
|   +-- supabase/
|       +-- client.ts                Browser Supabase client factory (public URL + anon key).
|       +-- server.ts                Server Supabase client factory with cookie-based session handling.
|       +-- proxy.ts                 Session refresh and auth redirect logic used by the root proxy.
|       +-- index.ts                 Barrel re-exporting the clients and updateSession.
|
+-- types/                          Generated and app-level TypeScript types.
|   +-- supabase.ts                  Auto-generated Supabase Database types.
|   +-- model-recommendations.ts     App-level interfaces for technical and fundamental recommendation rows.
|   +-- quarterly.ts                 Interface for a quarterly_fundamentals row.
|
+-- public/                         Static assets (brand logo).
```

Grouped notes:

- CSS Modules (`*.module.css`): each interactive component has a co-located
scoped stylesheet for color, layout, and animation. They are not listed
individually.
- Barrel files (`index.ts`): re-export a folder's public components/hooks for
cleaner imports.
- `app/(main)/portfolio/components/`: contains `PortfolioInsights.tsx` (tab
container), `PerformanceChart.tsx`, `CompositionChart.tsx`,
`PortfolioSwitcher.tsx`, `TransactionLedger.tsx`,
`PortfolioTransactionLedger.tsx`, and `QuickTradeModal.tsx` (each with a
matching CSS Module).
- `components/stocks/charts/`: contains `QuarterlyChartsPanel.tsx` (tab
container), `QuarterlyChartCard.tsx` (wrapper), `RevenueChart.tsx`,
`NetIncomeChart.tsx`, `EpsChart.tsx`, `BalanceSheetChart.tsx`, and a
`shared.ts` of formatting helpers.

### 1b. Model Training (`models/` and `historic-data/`)

```
historic-data/
+-- data.py                          One-shot script that downloads full S&P 500 daily OHLCV history from yfinance to a local CSV.
+-- initial_load.py                  Bulk-uploads the local OHLCV CSV into Supabase historic_data in retried, resumable batches.
+-- update_data.py                   Incremental daily updater: fetches new rows from yfinance and upserts into Supabase historic_data.

models/
|
+-- README.md                        Team guide for the technical PatchTST pipeline (Kaggle workflow, local setup, inference, automation).
+-- kaggle_cli_setup.md              Troubleshooting notes for Kaggle CLI version, Python 3.11, and GPU accelerator flags.
|
+-- data_raw/                        Training input data and prep.
|   +-- prune_quarterly_fundamentals.py  Filters/normalizes raw Compustat quarterly data to the OHLCV ticker universe.
|   +-- dataset-metadata.json        Kaggle dataset manifest used when versioning the raw OHLCV dataset.
|   +-- (large CSVs)                 Long-format OHLCV panel and quarterly fundamentals exports (contents not enumerated).
|
+-- patchtst_lib/                    Shared library imported by every notebook, inference, and backtest script.
|   +-- __init__.py                  Package marker / docstring for the shared PatchTST utilities.
|   +-- classification_head.py       Defines PatchTSTClassifier (encoder + multi-day 3-class head + optional industry embeddings).
|   +-- labeling.py                  Converts future price moves into down/flat/up labels (fixed-percent, rolling-vol, or ATR rules).
|   +-- training.py                  HuggingFace compute_metrics callback (accuracy, macro-F1, per-day accuracy).
|   +-- technical/
|   |   +-- features.py              Normalizes OHLCV columns and applies per-ticker scaling with log-volume transform.
|   |   +-- dataset.py               Builds the sliding-window classification dataset with industry IDs and chronological splits.
|   |   +-- backtest.py              Daily paper-trading backtest selecting top-N confident "up" picks (plus __init__.py).
|   +-- fundamental/
|       +-- config.py                Factory for a compact PatchTSTConfig tuned for 12-quarter fundamental windows.
|       +-- features.py              Normalized quarterly ratio/growth features with winsorization and z-scoring.
|       +-- dataset.py               Builds the quarterly dataset with publish-lag and forecast-lag to avoid lookahead bias.
|       +-- backtest.py              Quarterly paper-trading backtest (plus __init__.py).
|
+-- notebook_model_runs/             Active Kaggle training sandboxes (one per pipeline).
|   +-- technical/
|   |   +-- patchtst-technical-classifier.ipynb  Main technical training notebook (load data, train, evaluate, backtest, save).
|   |   +-- kernel-metadata.json     Kaggle kernel push config (dataset link, T4 GPU settings).
|   |   +-- requirements.txt         Python dependencies for local/Kaggle parity (PyTorch installed separately).
|   |   +-- pull_results.sh          Downloads Kaggle kernel outputs (checkpoint/, save_dir/, predictions, ticker map).
|   |   +-- backtest_local.py        Offline weighting-strategy comparison using pulled predictions (no model reload).
|   |   +-- ticker_industry.json     Ticker-to-industry mapping built during training for industry embeddings.
|   +-- fundamental/
|       +-- patchtst-fundamental-classifier.ipynb  Trains a compact PatchTST on quarterly fundamentals for next-quarter direction.
|       +-- kernel-metadata.json     Kaggle kernel config attaching both OHLCV and fundamentals datasets.
|       +-- requirements.txt         Same core ML stack as the technical pipeline.
|       +-- pull_results.sh          Pulls fundamental checkpoint/, save_dir_fund/, and predictions from Kaggle.
|       +-- backtest_local.py        Single-period quarterly backtest comparing weighting schemes.
|       +-- upload_recommendations.py  Uploads latest-quarter fundamental predictions to the fundamental_recommendations table.
|
+-- notebook_best_models/            Canonical "production" copies promoted from the best Kaggle runs.
    +-- technical/
    |   +-- patchtst-technical-classifier.ipynb  Frozen best technical notebook (with executed outputs).
    |   +-- local_inference.py       Production inference: load weights, fetch Supabase data, generate 5-day BUY/HOLD/SELL, optionally upsert.
    |   +-- ticker_industry.json     Industry map used at inference time.
    |   +-- save_dir/                Committed model artifacts: pytorch_model.bin, config.json, training_metadata.json (contents not enumerated).
    +-- fundamental/
        +-- patchtst-fundamental-classifier.ipynb  Frozen best fundamental notebook (with executed outputs).
```

Grouped notes:

- `checkpoint/`, `save_dir/`, `save_dir_fund/`, and `.venv/` hold training
artifacts and environments; they are gitignored or noted at the directory
level rather than enumerated.
- Both pipelines deliberately share `patchtst_lib/` so the notebooks, inference,
and backtest scripts never duplicate feature, labeling, model, or metric
logic.

### 1c. Docs (`docs/`)

For the purposes of this document, the relevant files in `docs/` are the two
SQL bootstrap scripts. Broader Supabase architecture (table dictionary,
relationships, and diagrams) is covered in a separate dedicated document.

```
docs/
+-- model_recommendations.sql        One-time Supabase setup: creates the model_recommendations table (short-horizon technical predictions), indexes, RLS, and a public read-only policy.
+-- fundamental_recommendations.sql  One-time Supabase setup: creates the fundamental_recommendations table (quarterly fundamental predictions), indexes, RLS, and a public read-only policy.
```

### 1d. GitHub Actions (`.github/workflows/`)

```
.github/workflows/
+-- daily-model-recommendations.yml  Scheduled workflow that refreshes market data and republishes technical recommendations to Supabase.
```

---

## 2. File Interactions

This section describes how the files above work together, why, and what user-facing or operational feature each interaction creates.

### 2a. Finance Dashboard

---

Authentication and route protection:

On nearly every request, the root `proxy.ts` calls `lib/supabase/proxy.ts` to refresh the Supabase session cookies and enforce routing rules (unauthenticated users are sent to `/sign-in`; signed-in users are kept out of `/` and `/sign-in`). Sign-in itself starts in `(auth)/sign-in/SignInButton.tsx` (Google OAuth via the browser client), and `(auth)/auth/callback/route.ts` completes the exchange, creates a `profiles` row if one does not exist, and redirects to `/dashboard`. The
`(main)/layout.tsx` acts as a second gate.

Feature:  secure, persistent Google sign-in with protected app routes and automatic profile creation.

---

Read path (server pages):

Each authenticated page follows a server-fetch then client-render pattern. A `page.tsx` (server component) creates a Supabase server client via `lib/supabase/server.ts`, loads the user's data (portfolios, holdings, watchlist, quarterly fundamentals), and passes it as props to a client component (for example `DashboardPage.tsx`, `WatchlistPage.tsx`, or`portfolio/[id]/page.tsx`).

Feature: fast first paint with server-rendered, authorized data.

---

Write path (server actions):

All mutations go through server actions rather than client-side database calls. `(main)/actions/portfolio.ts` handles portfolio CRUD and buy/sell trades (pricing trades via Finnhub and computing historical performance with Yahoo Finance), and `watchlist/actions.ts` handles watchlist changes. After writing, actions call `revalidatePath` so the server-rendered
pages refresh.

Feature: multi-portfolio management and trading, plus watchlist editing, with the UI staying in sync.

---

Live data via API route handlers:

Client components that need fresh data on demand call internal endpoints under `app/api/`. `stocksearch/route.ts` and
`stockquote/route.ts` proxy Finnhub (keeping the API key server-side), while `recommendations/route.ts` and `fundamental-recommendations/route.ts` read the latest rows from the Supabase `model_recommendations` and `fundamental_recommendations` tables. `DashboardPage.tsx` consumes both recommendation endpoints and lets the user toggle between Technical and Fundamental views. 

Feature: global stock search, live quotes, and the AI prediction panels.

---

Embedded market visuals: 

The `components/tradingview/` components render TradingView widgets for charts, heatmaps, company profiles, and fundamentals, configured by `widget-configs.ts`.

Feature: rich market overview and stock detail visuals without building charts from scratch.

---

Cross-cutting state:

`SelectedPortfolioContext` remembers which portfolio the user last viewed (persisted in localStorage) and is used by the navbar's portfolio link, `BuyAndWatchlist.tsx`, and `PortfolioSwitcher.tsx`. `ThemeContext` tracks dark/light mode.  

Feature: consistent "current portfolio" behavior across the app and a themeable UI.

---

### 2b. Model Training

---

Data acquisition:

`historic-data/data.py` and `initial_load.py` perform the one-time bulk load of S&P 500 daily OHLCV into the Supabase `historic_data` table. Thereafter, `update_data.py` queries the latest stored date and upserts only new business days from yfinance (on the `(date, ticker)` key).

Feature: a continuously maintained market-data store that both training snapshots and live inference read from.

---

Training on Kaggle: 

The notebooks in `notebook_model_runs/technical/` and `.../fundamental/` import `patchtst_lib/` for everything substantive: feature engineering and scaling (`features.py`), windowed datasets (`dataset.py`), label construction (`labeling.py`), the model itself (`classification_head.py`), and evaluation metrics (`training.py`). The technical pipeline uses a 128-day context to predict each of the next 5 trading days; the fundamental pipeline uses 12 quarters of normalized Compustat features with publish/forecast lags to predict the next quarter's direction. Each run writes checkpoints and a `save_dir` of weights, config, and metadata.

Feature: reproducible GPU training for both a technical (price) and a fundamental (financials) recommender, sharing one code library.

---

Artifact promotion: 

After a run, `pull_results.sh` downloads the Kaggle outputs locally; the best run is then copied into `notebook_best_models/` and committed to git, which is what makes a specific model "production." 

Feature: a clear, version-controlled "current best model" that automation can rely on.

---

Inference and publishing: 

`notebook_best_models/technical/local_inference.py` loads the committed weights, `config.json`, and `training_metadata.json`, fetches recent OHLCV from Supabase, reproduces the training-time scaling, attaches industry IDs from `ticker_industry.json`, runs the forward pass, and maps the output classes to SELL / HOLD / BUY for forecast days 1 to 5. With `--write-supabase` it upserts into `model_recommendations` (key`(ticker, context_end, forecast_day)`). On the fundamental side, `upload_recommendations.py` pushes the latest-quarter predictions into `fundamental_recommendations`.

Feature: the daily technical picks and the quarterly fundamental picks that the dashboard displays.

---

### 2c. Docs

The two SQL files are one-time bootstrap scripts that must be run in the Supabase SQL editor before the model pipeline can publish. `model_recommendations.sql` must exist before `local_inference.py --write-supabase` (and the GitHub Actions job) can upsert technical predictions, and `fundamental_recommendations.sql` must exist before `upload_recommendations.py` can upload fundamental predictions. Both scripts also enable Row Level Security with a public read-only policy, which is what lets the dashboard's API routes read the tables with the anon key while writes still require the service role key. 

Feature: the database tables and access rules that connect model output to the dashboard. (Full Supabase architecture is documented separately.)

---

### 2d. GitHub Actions / Workflow

`daily-model-recommendations.yml` ties data acquisition, inference, and Supabase together on a schedule (08:00 UTC, Tuesday to Saturday, after the prior US market day settles) and on manual dispatch. Each run: validates that the committed model artifacts exist, runs `historic-data/update_data.py --no-csv` to refresh Supabase market data, runs `local_inference.py --sp500 tickers --write-supabase --device cpu` to regenerate and publish technical recommendations, and uploads the resulting CSV as a build artifact. It reads its Supabase URL and service-role key from repository secrets. 

Feature: hands-off daily refresh so the dashboard's technical predictions stay
current without anyone running scripts manually.

---

## 3. End-to-End Data Flow

```
                          [ External market data ]
                  Finnhub        Yahoo Finance        yfinance
                     |                |                  |
                     |                |                  v
                     |                |        historic-data/update_data.py
                     |                |                  |
                     |                |                  v
                     |                |        +------------------------------+
                     |                |        |   Supabase (PostgreSQL)      |
                     |                |        |                              |
                     |                |        |  historic_data               |
                     |                |        |  model_recommendations       |
                     |                |        |  fundamental_recommendations |
                     |                |        |  profiles / portfolios /     |
                     |                |        |  watchlists / stocks /       |
                     |                |        |  quarterly_fundamentals      |
                     |                |        +------------------------------+
                     |                |               ^             |
   GitHub Actions    v                v   (write)     |             |  (read)
   (daily cron) -----+----------------+--> local_inference.py       |
                                       upload_recommendations.py    |
                                                                    |
                                                                    v
                                          +---------------------------------+
                                          |  Finance Dashboard (Next.js)    |
   live quotes / search  -------------->  |  server pages + server actions  |
   (Finnhub, Yahoo)                       |  + /api route handlers          |
                                          +---------------------------------+
                                                                    |
                                                                    v
                                                             [ User browser ]
```

Narrative: the model pipeline and the dashboard never talk to each other
directly. Supabase is the shared contract between them. The training/inference
side writes recommendations (and the data-update job keeps `historic_data`
fresh); the dashboard reads those tables through its API routes and renders
them, while separately pulling live quotes and search results from Finnhub and
Yahoo Finance for real-time interactivity.

---

## 4. Environment Variables and Secrets

Secret values are never committed. The dashboard reads from
`finance-dashboard/.env.local` (gitignored); the automation reads from GitHub
repository secrets. Only variable names are listed here.


| Variable                        | Used by                                          | Purpose                                                                                                       |
| ------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `NEXT_PUBLIC_SUPABASE_URL`      | Dashboard (client + server)                      | Supabase project URL.                                                                                         |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Dashboard (client + server)                      | Public anon key for RLS-protected reads/writes.                                                               |
| `SUPABASE_SERVICE_ROLE_KEY`     | Dashboard (server only), Model pipeline, Actions | Privileged key for service-role reads/writes (for example quarterly fundamentals and recommendation upserts). |
| `SUPABASE_URL`                  | Model pipeline, Actions                          | Supabase project URL for Python scripts.                                                                      |
| `SUPABASE_RECOMMENDATIONS_KEY`  | Model pipeline                                   | Write-capable key used by `local_inference.py --write-supabase` (set to the service role key).                |
| `FINNHUB_API_KEY`               | Dashboard (server)                               | Stock search, quotes, and trade pricing.                                                                      |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Training workflow / Kaggle CLI                   | Authenticate Kaggle dataset and kernel commands.                                                              |


---

## 5. Where to Start (Quick Reference)


| If you need to...              | Start here                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| Add a protected page           | `finance-dashboard/app/(main)/` and check `publicRoutes` in `lib/supabase/proxy.ts`. |
| Read user data on load         | A server `page.tsx` using `createClient()` from `lib/supabase/server.ts`.            |
| Mutate the database            | A server action in `app/(main)/actions/` or `app/(main)/watchlist/actions.ts`.       |
| Add a client-callable endpoint | `app/api/<name>/route.ts`.                                                           |
| Regenerate Supabase types      | `npm run update-db-types` (updates `types/supabase.ts`).                             |
| Refresh market data            | `historic-data/update_data.py`.                                                      |
| Train a model                  | `models/notebook_model_runs/<pipeline>/` plus `models/README.md`.                    |
| Run production inference       | `models/notebook_best_models/technical/local_inference.py`.                          |
| Understand the model code      | `models/patchtst_lib/`.                                                              |
| Upload fundamental picks       | `models/notebook_model_runs/fundamental/upload_recommendations.py`.                  |
| Change the daily automation    | `.github/workflows/daily-model-recommendations.yml`.                                 |


---

## 6. Known Gaps and Notes

- Uninitialized page routes (i.e. `/settings`) may be found and are part of future work.
- The fundamental recommendation upload is currently a manual step (`upload_recommendations.py`); only the technical recommendations are refreshed automatically by GitHub Actions.

