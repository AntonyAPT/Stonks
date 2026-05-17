from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from supabase import create_client


DEFAULT_SUPABASE_URL = "https://jnjpfkbdaoxuumayxivh.supabase.co"
DEFAULT_SUPABASE_KEY = "sb_publishable_-JvliZ3nQWSWVD9UTmP6xw_j9BDFaFd"
DEFAULT_TABLE_NAME = "historic_data"
DEFAULT_CSV_PATH = "sp500_daily_max.csv"
DEFAULT_BATCH_SIZE = 500
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append new S&P 500 daily rows to Supabase.")
    parser.add_argument("--csv-path", type=Path, default=Path(os.getenv("HISTORIC_CSV_PATH", DEFAULT_CSV_PATH)))
    parser.add_argument("--no-csv", action="store_true", help="Do not read or update a local CSV file.")
    parser.add_argument("--start-date", help="Optional YYYY-MM-DD override for the first date to download.")
    parser.add_argument("--table", default=os.getenv("SUPABASE_HISTORIC_TABLE", DEFAULT_TABLE_NAME))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("SUPABASE_BATCH_SIZE", DEFAULT_BATCH_SIZE)))
    parser.add_argument(
        "--supabase-url",
        default=first_env("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL", default=DEFAULT_SUPABASE_URL),
    )
    parser.add_argument(
        "--supabase-key",
        default=first_env(
            "SUPABASE_HISTORIC_DATA_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
            "SUPABASE_RECOMMENDATIONS_KEY",
            "SUPABASE_KEY",
            "NEXT_PUBLIC_SUPABASE_ANON_KEY",
            default=DEFAULT_SUPABASE_KEY,
        ),
    )
    return parser.parse_args()


def fetch_sp500_tickers_and_sectors() -> tuple[list[str], dict[str, str]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(SP500_URL, headers=headers, timeout=30)
    response.raise_for_status()
    table = pd.read_html(response.text)[0]
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    sector_map = dict(zip(table["Symbol"], table["GICS Sector"]))
    return list(sector_map.keys()), sector_map


def latest_supabase_date(client, table_name: str) -> pd.Timestamp | None:
    response = (
        client.table(table_name)
        .select("date")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return None
    return pd.to_datetime(rows[0]["date"])


def latest_csv_date(csv_path: Path) -> pd.Timestamp | None:
    if not csv_path.exists():
        return None
    header = pd.read_csv(csv_path, nrows=0)
    date_column = next((column for column in header.columns if str(column).lower() == "date"), None)
    if date_column is None:
        raise ValueError(f"{csv_path} does not contain a date column.")
    frame = pd.read_csv(csv_path, usecols=[date_column], parse_dates=[date_column])
    if frame.empty:
        return None
    return pd.to_datetime(frame[date_column].max())


def download_new_rows(tickers: list[str], sector_map: dict[str, str], start_date: date, end_date: date) -> pd.DataFrame:
    print(f"Fetching new data from {start_date} to {end_date}...")
    raw = yf.download(
        tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame()

    frame = raw.stack(level=0, future_stack=True).reset_index()
    frame.columns.name = None
    frame.rename(columns={"level_1": "Ticker"}, inplace=True)
    frame["Sector"] = frame["Ticker"].map(sector_map)
    frame.columns = [str(column).lower() for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame.dropna(subset=["open", "high", "low", "close", "volume"], how="all", inplace=True)
    frame = frame.where(pd.notnull(frame), None)
    return frame


def update_csv(csv_path: Path, new_rows: pd.DataFrame) -> None:
    if not csv_path.exists():
        print(f"CSV {csv_path} not found; skipping local CSV update.")
        return

    print(f"Appending {len(new_rows):,} new rows to {csv_path}...")
    existing = pd.read_csv(csv_path)
    existing.columns = [str(column).lower() for column in existing.columns]
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    combined.drop_duplicates(subset=["date", "ticker"], keep="last", inplace=True)
    combined.sort_values(["ticker", "date"], inplace=True)
    combined.to_csv(csv_path, index=False)
    print(f"CSV updated to {combined['date'].max()}")


def upsert_rows(client, table_name: str, rows: list[dict], batch_size: int) -> None:
    print(f"Pushing {len(rows):,} new row(s) to Supabase table '{table_name}'...")
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        client.table(table_name).upsert(batch, on_conflict="date,ticker").execute()
        print(f"  Upserted rows {start + 1:,}-{start + len(batch):,} of {len(rows):,}")


def main() -> None:
    args = parse_args()
    client = create_client(args.supabase_url, args.supabase_key)

    csv_last_date = None if args.no_csv else latest_csv_date(args.csv_path)
    supabase_last_date = latest_supabase_date(client, args.table)
    if args.start_date:
        last_date = pd.to_datetime(args.start_date) - pd.Timedelta(days=1)
    else:
        known_dates = [known_date for known_date in [csv_last_date, supabase_last_date] if known_date is not None]
        if not known_dates:
            raise SystemExit(
                "No existing CSV or Supabase rows were found. Provide --start-date YYYY-MM-DD "
                "for an initial automated load."
            )
        last_date = max(known_dates)

    start_date = (last_date + timedelta(days=1)).date()
    end_date = datetime.today().date() + timedelta(days=1)

    print(f"Last CSV date: {csv_last_date.date() if csv_last_date is not None else 'not used'}")
    print(f"Last Supabase date: {supabase_last_date.date() if supabase_last_date is not None else 'none'}")

    if start_date >= end_date:
        print("Already up to date; nothing to fetch.")
        return

    tickers, sector_map = fetch_sp500_tickers_and_sectors()
    new_rows = download_new_rows(tickers, sector_map, start_date, end_date)
    if new_rows.empty:
        print("No new data returned; market may have been closed since the last update.")
        return

    if not args.no_csv:
        update_csv(args.csv_path, new_rows)

    upsert_rows(client, args.table, new_rows.to_dict(orient="records"), args.batch_size)
    print("Done. Historic data is up to date.")


if __name__ == "__main__":
    main()
