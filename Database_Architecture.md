# STONKS - Database Architecture

This document explains how the STONKS project uses Supabase as its shared backend. It focuses on the database architecture, the main table groups, how those tables connect to the Next.js dashboard and Python model pipeline, and how authentication and permissions shape the data flow.

Supabase provides three important backend services for this project:

1. PostgreSQL database storage for user data, stock reference data, historic market data, quarterly fundamentals, and model recommendations.
2. Supabase Auth for Google sign-in, session cookies, and user identity.
3. Row Level Security and API keys for separating public reads, authenticated user actions, and privileged server or automation writes.

The dashboard and model pipeline do not call each other directly. Supabase is the contract between them: the dashboard reads and writes user-facing data, while the data and model scripts keep the market-data and recommendation tables fresh.

---

## 1. High-Level Architecture

```
                         +--------------------------+
                         |   External Data Sources  |
                         | Finnhub, Yahoo, yfinance |
                         +------------+-------------+
                                      |
                                      v
       +------------------+   +----------------------+   +----------------------+
       | Python Ingestion |-->| Supabase PostgreSQL  |<--| Next.js Dashboard    |
       | and ML Pipeline  |   | + Auth + RLS         |   | Server Components    |
       +------------------+   +----------------------+   | Server Actions       |
                                      ^                  | API Route Handlers   |
                                      |                  +----------------------+
                                      |
                         +------------+-------------+
                         | GitHub Actions / Manual  |
                         | Model Upload Scripts     |
                         +--------------------------+
```

Supabase sits in the middle of the application. The main data zones are:

- Identity data: Supabase Auth plus the `profiles` table.
- User-owned containers: `portfolios` and `watchlists`.
- User-owned records: `portfolio_items` and `watchlist_items`.
- Shared reference data: `stocks`.
- Market and fundamentals data: `historic_data` and `quarterly_fundamentals`.
- Model output: `model_recommendations` and `fundamental_recommendations`.

---

## 2. Database Zones and Tables

### 2a. Identity Zone

`auth.users`

Managed by Supabase Auth. It stores login identity, OAuth provider information, and the canonical user UUID. Application code does not query or mutate this table directly for normal features.

`profiles`

Stores app-level user information:

- `id`: matches the Supabase Auth user ID.
- `username`: display name, usually taken from Google profile metadata.
- `avatar_url`: Google avatar URL.
- `created_at`: profile creation timestamp.

When a user completes Google OAuth, `finance-dashboard/app/(auth)/auth/callback/route.ts` exchanges the OAuth code for a Supabase session and makes sure a matching `profiles` row exists. This table is the parent for user-owned portfolios and watchlists.

### 2b. User Container Zone

`portfolios`

Represents named collections of holdings owned by one user.

Important fields:

- `id`: portfolio UUID.
- `user_id`: references `profiles.id`.
- `name`: user-visible portfolio name.
- `is_default`: marks the user's default portfolio.
- `created_at`: creation timestamp.

Users can have many portfolios. The default portfolio gives the app a stable place to send a new user and gives the navbar a fallback when no specific portfolio is selected.

`watchlists`

Represents named collections of stocks the user wants to monitor.

Important fields:

- `id`: watchlist UUID.
- `user_id`: references `profiles.id`.
- `name`: user-visible watchlist name.
- `is_default`: marks the user's default watchlist.

The current app mainly uses one default watchlist per user. `watchlist/actions.ts` creates it on demand if the user adds a ticker before a default watchlist exists.

### 2c. User Record Zone

`portfolio_items`

Stores portfolio transactions. This table behaves like a transaction ledger rather than a table of only current holdings.

Important fields:

- `id`: transaction UUID.
- `portfolio_id`: references `portfolios.id`.
- `stock_ticker`: references `stocks.ticker`.
- `quantity`: number of shares.
- `buy_price`: transaction unit price. The name is reused for both buy and sell rows.
- `buy_date`: transaction timestamp. The name is reused for both buy and sell rows.
- `transaction_type`: usually `buy` or `sell`.

The dashboard computes current holdings by replaying these rows in chronological order. Buys increase share count and sells decrease it. This is why the portfolio page can show transaction history, cost basis, remaining open lots, unrealized gain, and holdings from a single table.

`watchlist_items`

Stores tickers inside a watchlist.

Important fields:

- `id`: watchlist item UUID.
- `watchlist_id`: references `watchlists.id`.
- `stock_ticker`: references `stocks.ticker`.
- `added_at`: timestamp used for ordering.

The app prevents duplicate watchlist entries by checking for an existing row with the same `watchlist_id` and `stock_ticker` before inserting.

### 2d. Shared Reference Zone

`stocks`

Stores shared ticker metadata used by portfolios and watchlists.

Important fields:

- `ticker`: primary key.
- `company_name`: display name.
- `industry`: Finnhub industry string.
- `last_updated`: optional refresh timestamp.

This table avoids repeated external profile calls for every dashboard render. Server actions upsert or insert stock rows when a user adds a stock to a portfolio or watchlist.

### 2e. Market and Fundamentals Zone

`historic_data`

Stores daily S&P 500 OHLCV data used by the technical model.

Important fields:

- `date`: trading date.
- `ticker`: ticker symbol.
- `open`, `high`, `low`, `close`: daily adjusted price fields.
- `volume`: daily volume.
- `sector`: GICS sector from the S&P 500 constituents list.

The historic data scripts treat `(date, ticker)` as the logical unique key. `historic-data/update_data.py` checks the latest date already stored in Supabase, downloads only newer business days with `yfinance`, and upserts rows back into `historic_data`.

`quarterly_fundamentals`

Stores quarterly company fundamentals used by stock detail pages and the fundamental model workflow.

Important fields:

- `tic`: ticker symbol.
- `datadate`: reporting date.
- `fyearq`, `fqtr`: fiscal year and quarter.
- `revtq`: revenue.
- `niq`: net income.
- `epspxq`: EPS.
- `atq`: assets.
- `ltq`: liabilities.
- `dlttq`, `dlcq`: long-term and current debt.

The dashboard reads this table through `finance-dashboard/lib/quarterly.ts`, which uses the service role key on the server. Stock detail pages then render tabular and chart views from these rows.

### 2f. Recommendation Zone

`model_recommendations`

Stores the technical PatchTST model's short-horizon predictions.

Important fields:

- `ticker`: predicted ticker.
- `sector`, `industry`: classification metadata.
- `context_start`, `context_end`: input window used for inference.
- `forecast_day`: day 1 through day 5.
- `forecast_date`: predicted trading date.
- `predicted_class`: numeric class, where the model uses down/flat/up.
- `predicted_direction`: `down`, `flat`, or `up`.
- `recommendation`: dashboard label, `SELL`, `HOLD`, or `BUY`.
- `confidence`, `prob_down`, `prob_flat`, `prob_up`: model probabilities.
- `last_close`: latest close at inference time.
- `run_timestamp`: inference run timestamp.

The table has a unique prediction key of `(ticker, context_end, forecast_day)`. `local_inference.py --write-supabase` upserts on that key so each new run replaces the latest prediction for the same ticker, context date, and forecast day.

`fundamental_recommendations`

Stores the quarterly fundamental model's predictions.

Important fields:

- `ticker`: predicted ticker.
- `context_start_quarter`, `context_end_quarter`: quarterly input window.
- `decision_date`: date when the recommendation becomes actionable.
- `forecast_end_date`: forecast horizon endpoint.
- `context_year`, `forecast_year`: model context and target years.
- `predicted_class`, `predicted_direction`, `recommendation`: model decision.
- `confidence`: model confidence.
- `actual_class`, `actual_direction`, `forward_return`: optional realized outcome fields for evaluation.
- `run_timestamp`: upload timestamp.

The table has a unique prediction key of `(ticker, context_end_quarter, forecast_year)`. `upload_recommendations.py` upserts on that key after a fundamental model run is pulled from Kaggle.

---

## 3. Main Relationships

The core relational shape is:

```
auth.users
   |
   | one-to-one
   v
profiles
   |
   +-- one-to-many --> portfolios
   |                    |
   |                    +-- one-to-many --> portfolio_items
   |                                         |
   |                                         +-- many-to-one --> stocks
   |
   +-- one-to-many --> watchlists
                        |
                        +-- one-to-many --> watchlist_items
                                             |
                                             +-- many-to-one --> stocks

historic_data              independent shared ML input table
quarterly_fundamentals     independent shared fundamentals table
model_recommendations      independent shared technical output table
fundamental_recommendations independent shared fundamental output table
```

The recommendation and market-data tables are intentionally not tied to user IDs. They are shared application data. User-specific records are separated into portfolios, watchlists, and their item tables.

---

## 4. How the Dashboard Connects to Supabase

### 4a. Supabase Client Factories

`finance-dashboard/lib/supabase/client.ts`

Creates the browser Supabase client with:

- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`

This is used for browser-side auth tasks such as starting Google sign-in and signing out.

`finance-dashboard/lib/supabase/server.ts`

Creates the server Supabase client with cookie support. Server Components, API routes, and Server Actions use this client to identify the current user and query Supabase with the user's session.

`finance-dashboard/lib/supabase/proxy.ts`

Refreshes Supabase session cookies and protects routes. It redirects unauthenticated visitors to `/sign-in` and redirects authenticated users away from public auth routes toward `/dashboard`.

`finance-dashboard/lib/quarterly.ts`

Creates a server-only Supabase client using `SUPABASE_SERVICE_ROLE_KEY`. This is used for privileged reads from `quarterly_fundamentals`. Because the service role bypasses RLS, this helper is marked `server-only` and is never imported into client components.

### 4b. Authentication Flow

1. The user clicks the Google sign-in button.
2. The browser Supabase client starts OAuth.
3. Supabase redirects back to `/auth/callback`.
4. The callback route exchanges the code for a Supabase session.
5. The callback route ensures a row exists in `profiles`.
6. The user is redirected to `/dashboard`.
7. The root proxy and `(main)/layout.tsx` protect future authenticated routes.

The user's Supabase Auth ID becomes the key used by `profiles`, `portfolios`, and `watchlists`.

### 4c. Read Path

Most user-facing pages are Server Components. They create a Supabase server client, call `supabase.auth.getUser()`, and then query only the current user's records.

Examples:

- `dashboard/page.tsx` reads the default watchlist and all portfolio transactions for dashboard summaries.
- `portfolios/page.tsx` reads all portfolios and joins `portfolio_items` to compute cost basis.
- `portfolio/[id]/page.tsx` verifies the portfolio belongs to the current user, reads its transactions, fetches matching `stocks` metadata, and computes holdings.
- `watchlist/page.tsx` reads the default watchlist, its items, and matching company names from `stocks`.
- `stocks/[ticker]/page.tsx` reads `quarterly_fundamentals` for the selected ticker.

This keeps sensitive queries on the server and sends only the rendered page data to the browser.

### 4d. Write Path

User mutations are handled through Server Actions:

- `app/(main)/actions/portfolio.ts`
- `app/(main)/watchlist/actions.ts`

The actions follow the same pattern:

1. Create a Supabase server client.
2. Get the authenticated user.
3. Verify the user owns the portfolio or watchlist being changed.
4. Fetch live pricing or profile data from Finnhub when needed.
5. Insert, update, delete, or upsert Supabase rows.
6. Call `revalidatePath()` so server-rendered pages refresh.

This is used for creating, renaming, and deleting portfolios; buying and selling stocks; creating a default watchlist; adding watchlist items; and removing watchlist items.

### 4e. API Route Reads

Some client components need fresh data without a full page navigation. The dashboard exposes internal API routes:

- `/api/recommendations`: reads the newest `model_recommendations` context and returns the top technical predictions.
- `/api/fundamental-recommendations`: reads the newest `fundamental_recommendations` quarter and returns the top fundamental predictions.
- `/api/quarterly`: wraps `getQuarterlyFundamentals()` for ticker-specific fundamentals.
- `/api/stocksearch`: proxies Finnhub search.
- `/api/stockquote`: proxies Finnhub live quotes.

The market-data API routes keep `FINNHUB_API_KEY` server-side. The recommendation routes query Supabase and return dashboard-friendly JSON.

---

## 5. How the Model Pipeline Connects to Supabase

### 5a. Historic Data Ingestion

`historic-data/initial_load.py`

Performs a one-time bulk upload from a local S&P 500 OHLCV CSV into `historic_data`. It uploads in batches and records progress so long uploads can resume.

`historic-data/update_data.py`

Performs recurring incremental updates:

1. Reads the latest stored date from `historic_data`.
2. Fetches current S&P 500 tickers and sectors.
3. Downloads newer daily OHLCV rows from `yfinance`.
4. Normalizes columns to the database shape.
5. Upserts rows on `(date, ticker)`.

The GitHub Actions workflow runs this before technical inference so the model sees fresh data.

### 5b. Technical Recommendations

`models/notebook_best_models/technical/local_inference.py`

This script is the production technical inference path:

1. Loads the promoted PatchTST model artifacts from `notebook_best_models/technical/save_dir`.
2. Reads recent OHLCV rows from `historic_data`.
3. Normalizes the data to match training.
4. Builds the latest inference window for each ticker.
5. Runs the model for forecast days 1 through 5.
6. Converts model classes into `SELL`, `HOLD`, and `BUY`.
7. Writes a local CSV.
8. With `--write-supabase`, upserts rows into `model_recommendations`.

The dashboard does not run the model. It only reads the rows this script publishes.

### 5c. Fundamental Recommendations

`models/notebook_model_runs/fundamental/upload_recommendations.py`

This script uploads the latest fundamental model predictions:

1. Reads the pulled Kaggle prediction CSV.
2. Keeps the latest `context_end_quarter` by default.
3. Normalizes dates, ticker symbols, classes, and confidence.
4. Adds the dashboard recommendation label.
5. Upserts rows into `fundamental_recommendations`.

This upload is currently a manual step after pulling fundamental model results.

---

## 6. Security and Permissions

Supabase access is split by key and execution environment.

`NEXT_PUBLIC_SUPABASE_ANON_KEY`

Used by the browser client and the normal server client. It is safe to expose because database access should be controlled by Supabase Auth and Row Level Security. User-facing reads and writes should be scoped to the signed-in user's session.

`SUPABASE_SERVICE_ROLE_KEY`

Used only on trusted servers, local scripts, and GitHub Actions. It bypasses RLS and can read or write privileged tables. It should never be exposed in client code.

Recommendation table policies:

- `model_recommendations.sql` enables RLS and allows public read access to `anon` and `authenticated`.
- `fundamental_recommendations.sql` enables RLS and allows public read access to `anon` and `authenticated`.
- Neither table has public insert or update policies. Writes require the service role key or another write-capable secret used by automation.

Application-level safeguards:

- Server pages and server actions call `supabase.auth.getUser()` before using user data.
- Portfolio actions verify `portfolios.user_id` before mutating portfolio rows.
- The portfolio detail page returns `notFound()` if the requested portfolio does not belong to the current user.
- Watchlist actions operate on the authenticated user's default watchlist.
- Finnhub keys and service-role Supabase keys stay on the server.

---

## 7. Environment Variables

| Variable | Used by | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_SUPABASE_URL` | Dashboard client and server | Supabase project URL. |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Dashboard client and server | Public key for Auth and RLS-controlled database access. |
| `SUPABASE_SERVICE_ROLE_KEY` | Server-only dashboard helpers, scripts, GitHub Actions | Privileged Supabase key for service-role reads and writes. |
| `SUPABASE_URL` | Python scripts and automation | Supabase project URL outside the Next.js app. |
| `SUPABASE_RECOMMENDATIONS_KEY` | Model upload scripts | Write-capable key for recommendation upserts. Usually the service role key. |
| `SUPABASE_HISTORIC_DATA_KEY` | Historic data updater | Optional write-capable key for `historic_data` upserts. |
| `SUPABASE_HISTORIC_TABLE` | Historic data updater | Optional table override, defaults to `historic_data`. |
| `SUPABASE_FUNDAMENTAL_RECOMMENDATIONS_TABLE` | Fundamental upload script | Optional table override, defaults to `fundamental_recommendations`. |
| `FINNHUB_API_KEY` | Dashboard server routes and actions | Stock search, live quotes, profile data, and trade pricing. |

---

## 8. Operational Notes

- Regenerate TypeScript database types with `npm run update-db-types` from `finance-dashboard/` when the Supabase schema changes.
- Run `docs/model_recommendations.sql` once before publishing technical predictions.
- Run `docs/fundamental_recommendations.sql` once before publishing fundamental predictions.
- Use `historic-data/update_data.py --no-csv` for automated historic-data refreshes.
- Use `local_inference.py --write-supabase` only with a write-capable Supabase key.
- Keep service-role secrets in `.env.local` or GitHub repository secrets, never in browser-executed code.
- Treat `portfolio_items` as an append-style transaction ledger. Current holdings are derived, not stored as a separate snapshot.
- The recommendation tables are shared global data. They are not personalized to a user's portfolio or watchlist.

---

## 9. Summary

The database architecture uses Supabase as the backbone for both the application and the model pipeline. User identity and ownership are handled through Supabase Auth, `profiles`, portfolios, and watchlists. Market history and fundamentals are shared data sources. The Python model pipeline reads those sources and publishes recommendations. The Next.js dashboard reads the latest database state, performs user mutations through server actions, and uses internal API routes for fresh recommendation and market-data views.

In short: Supabase is the central integration layer that lets the finance dashboard, model training workflow, and automation system operate as one project without tightly coupling their code.
