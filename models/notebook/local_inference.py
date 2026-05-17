"""Run local PatchTST inference against recent historic_data rows.

Examples
--------
Smoke test a few tickers:

    python local_inference.py --tickers AAPL MSFT NVDA

Run the full S&P 500 universe from Supabase:

    python local_inference.py --all-tickers

The script writes a flat recommendation CSV. Pass ``--write-supabase`` after
the output looks sane to upsert those rows into Supabase for the website.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = NOTEBOOK_DIR / "save_dir" / "patchtst_cls_sector_all_11_sectors"
DEFAULT_OUTPUT_PATH = NOTEBOOK_DIR / "save_dir" / "latest_recommendations.csv"
DEFAULT_NOTEBOOK_PATH = NOTEBOOK_DIR / "patchtst-default.ipynb"
DEFAULT_INDUSTRY_MAP_PATH = NOTEBOOK_DIR / "ticker_industry.json"

DEFAULT_SUPABASE_URL = "https://jnjpfkbdaoxuumayxivh.supabase.co"
DEFAULT_SUPABASE_KEY = "sb_publishable_-JvliZ3nQWSWVD9UTmP6xw_j9BDFaFd"
DEFAULT_TABLE = "historic_data"
DEFAULT_RECOMMENDATIONS_TABLE = "model_recommendations"

TIMESTAMP_COLUMN = "Date"
TICKER_COLUMN = "Ticker"
SECTOR_COLUMN = "Sector"
TARGET_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CLASS_NAMES = {0: "down", 1: "flat", 2: "up"}
RECOMMENDATIONS = {"down": "SELL", "flat": "HOLD", "up": "BUY"}
torch = None
PatchTSTConfig = None
PatchTSTClassifier = None


def first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local PatchTST stock recommendations.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--input-csv", type=Path, help="Optional local history CSV instead of Supabase.")
    parser.add_argument("--tickers", nargs="*", help="Ticker symbols for a small local run.")
    parser.add_argument("--all-tickers", action="store_true", help="Infer every ticker found in the data source.")
    parser.add_argument(
        "--sp500-tickers",
        action="store_true",
        help="Fetch the current S&P 500 ticker list and query those tickers one at a time.",
    )
    parser.add_argument("--history-table", default=DEFAULT_TABLE)
    parser.add_argument("--recommendations-table", default=DEFAULT_RECOMMENDATIONS_TABLE)
    parser.add_argument(
        "--supabase-url",
        default=first_env("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL", default=DEFAULT_SUPABASE_URL),
    )
    parser.add_argument(
        "--supabase-key",
        default=first_env(
            "SUPABASE_RECOMMENDATIONS_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
            "SUPABASE_KEY",
            "NEXT_PUBLIC_SUPABASE_ANON_KEY",
            default=DEFAULT_SUPABASE_KEY,
        ),
    )
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument(
        "--max-rows-per-ticker",
        type=int,
        default=7000,
        help="Supabase row cap per ticker for ticker-specific inference fetches.",
    )
    parser.add_argument("--industry-map", type=Path, default=DEFAULT_INDUSTRY_MAP_PATH)
    parser.add_argument("--write-supabase", action="store_true", help="Upsert generated recommendations to Supabase.")
    parser.add_argument("--upsert-batch-size", type=int, default=500)
    parser.add_argument(
        "--build-industry-map",
        action="store_true",
        help="Use yfinance to build ticker_industry.json when it is missing.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def fetch_sp500_tickers() -> list[str]:
    try:
        import requests
    except ImportError as exc:
        raise SystemExit("Install requests first: pip install requests") from exc

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    table = pd.read_html(response.text)[0]
    tickers = table["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper()
    return sorted(tickers.unique().tolist())


def load_ml_dependencies() -> None:
    global torch, PatchTSTConfig, PatchTSTClassifier
    try:
        import torch as _torch
        from transformers import PatchTSTConfig as _PatchTSTConfig
        from classification_head import PatchTSTClassifier as _PatchTSTClassifier
    except ImportError as exc:
        raise SystemExit(
            "Install local model dependencies first. For macOS CPU/MPS, run something like:\n"
            "  pip install torch torchvision torchaudio\n"
            "  pip install -r models/notebook/requirements.txt"
        ) from exc

    torch = _torch
    PatchTSTConfig = _PatchTSTConfig
    PatchTSTClassifier = _PatchTSTClassifier


def normalize_historic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = {
        "date": TIMESTAMP_COLUMN,
        "ticker": TICKER_COLUMN,
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "sector": SECTOR_COLUMN,
    }
    rename = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in canonical:
            rename[column] = canonical[key]
    out = frame.rename(columns=rename).copy()
    missing = {TIMESTAMP_COLUMN, TICKER_COLUMN, *TARGET_COLUMNS}.difference(out.columns)
    if missing:
        raise ValueError(f"Historic data is missing required columns: {sorted(missing)}")
    if SECTOR_COLUMN not in out.columns:
        out[SECTOR_COLUMN] = "Unknown"
    out[TIMESTAMP_COLUMN] = pd.to_datetime(out[TIMESTAMP_COLUMN])
    out[TICKER_COLUMN] = out[TICKER_COLUMN].astype(str).str.upper()
    out[SECTOR_COLUMN] = out[SECTOR_COLUMN].astype(str)
    out = out.dropna(subset=TARGET_COLUMNS, how="any")
    out["Volume"] = np.log1p(out["Volume"].astype(float))
    return out.sort_values([TICKER_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)


def fetch_supabase_rows(args: argparse.Namespace) -> pd.DataFrame:
    try:
        from supabase import create_client
    except ImportError as exc:
        raise SystemExit("Install supabase first: pip install supabase") from exc

    client = create_client(args.supabase_url, args.supabase_key)
    columns = "date,ticker,open,high,low,close,volume,sector"

    if args.tickers:
        all_rows: list[dict] = []
        for ticker in [ticker.upper() for ticker in args.tickers]:
            response = (
                client.table(args.history_table)
                .select(columns)
                .eq("ticker", ticker)
                .order("date", desc=True)
                .limit(args.max_rows_per_ticker)
                .execute()
            )
            rows = response.data or []
            print(f"  {ticker}: fetched {len(rows):,} row(s)")
            all_rows.extend(rows)
        if not all_rows:
            raise ValueError("No historic rows returned from Supabase.")
        return pd.DataFrame(all_rows)

    all_rows: list[dict] = []
    start = 0
    while True:
        response = (
            client.table(args.history_table)
            .select(columns)
            .order("ticker")
            .order("date")
            .range(start, start + args.page_size - 1)
            .execute()
        )
        page = response.data or []
        all_rows.extend(page)
        if len(page) < args.page_size:
            break
        start += args.page_size

    if not all_rows:
        raise ValueError("No historic rows returned from Supabase.")
    return pd.DataFrame(all_rows)


def load_history(args: argparse.Namespace) -> pd.DataFrame:
    if args.sp500_tickers:
        args.tickers = fetch_sp500_tickers()
        print(f"Fetched {len(args.tickers):,} S&P 500 ticker(s).")

    if not args.all_tickers and not args.tickers:
        raise SystemExit(
            "Pass --tickers AAPL MSFT for a smoke test, --sp500-tickers for the S&P 500, "
            "or --all-tickers for the full source table."
        )

    if args.input_csv:
        raw = pd.read_csv(args.input_csv)
        if args.tickers:
            tickers = {ticker.upper() for ticker in args.tickers}
            ticker_col = next((c for c in raw.columns if str(c).lower() == "ticker"), "ticker")
            raw = raw[raw[ticker_col].astype(str).str.upper().isin(tickers)]
    else:
        raw = fetch_supabase_rows(args)
    return normalize_historic_columns(raw)


def compute_train_split_scaler(frame: pd.DataFrame, train_frac: float = 0.8) -> tuple[dict, pd.Series, pd.Series]:
    stats = {}
    train_parts = []
    for ticker, group in frame.groupby(TICKER_COLUMN, sort=False):
        group = group.sort_values(TIMESTAMP_COLUMN)
        train_end = max(1, int(len(group) * train_frac))
        train_group = group.iloc[:train_end]
        mean = train_group[TARGET_COLUMNS].astype(float).mean()
        std = train_group[TARGET_COLUMNS].astype(float).std().replace(0, 1.0).fillna(1.0)
        stats[str(ticker)] = (mean, std)
        train_parts.append(train_group)

    train_frame = pd.concat(train_parts, ignore_index=True)
    global_mean = train_frame[TARGET_COLUMNS].astype(float).mean()
    global_std = train_frame[TARGET_COLUMNS].astype(float).std().replace(0, 1.0).fillna(1.0)
    return stats, global_mean, global_std


def extract_training_industries(notebook_path: Path = DEFAULT_NOTEBOOK_PATH) -> list[str]:
    if not notebook_path.exists():
        return []
    try:
        notebook = json.loads(notebook_path.read_text())
    except Exception:
        return []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            match = re.search(r"Industries: (\[.*\])", text, re.S)
            if match:
                return list(ast.literal_eval(match.group(1)))
    return []


def build_industry_map(tickers: list[str], output_path: Path) -> dict[str, str]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("Install yfinance first or provide --industry-map.") from exc

    mapping = {}
    for ticker in tickers:
        try:
            mapping[ticker] = yf.Ticker(ticker).info.get("industry", "Unknown") or "Unknown"
        except Exception:
            mapping[ticker] = "Unknown"
    output_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    return mapping


def load_industry_inputs(args: argparse.Namespace, tickers: list[str], expected_count: int) -> tuple[dict[str, str], dict[str, int]]:
    industries = extract_training_industries()
    if industries and len(industries) != expected_count:
        print(
            f"Warning: notebook output has {len(industries)} industries, "
            f"but model metadata expects {expected_count}."
        )
    industry_to_id = {industry: idx for idx, industry in enumerate(industries)}

    if args.industry_map.exists():
        ticker_industry = json.loads(args.industry_map.read_text())
        ticker_industry = {str(k).upper(): str(v) for k, v in ticker_industry.items()}
    elif args.build_industry_map:
        print(f"Building {args.industry_map} with yfinance. This may take a while.")
        ticker_industry = build_industry_map(tickers, args.industry_map)
    else:
        ticker_industry = {}

    if expected_count > 0 and not industry_to_id:
        print("Warning: could not recover training industry ids; using industry_id=0 for all tickers.")
    elif expected_count > 0 and not ticker_industry:
        unknown_id = industry_to_id.get("Unknown", 0)
        print(
            "Warning: ticker_industry.json is missing; using the Unknown industry "
            f"embedding id ({unknown_id}) for every ticker."
        )
    return ticker_industry, industry_to_id


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_dir: Path, device: torch.device) -> tuple[PatchTSTClassifier, dict]:
    metadata_path = model_dir / "training_metadata.json"
    weights_path = model_dir / "pytorch_model.bin"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing model artifacts under {model_dir}")

    metadata = json.loads(metadata_path.read_text())
    config = PatchTSTConfig.from_pretrained(model_dir)
    model = PatchTSTClassifier(
        config=config,
        horizon=int(metadata["forecast_horizon"]),
        n_classes=3,
        class_weights=torch.ones(3),
        num_industries=int(metadata.get("num_industries", 0)),
        industry_embedding_dim=int(metadata.get("industry_embedding_dim", 8)),
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata


def make_latest_batch(
    frame: pd.DataFrame,
    metadata: dict,
    ticker_industry: dict[str, str],
    industry_to_id: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor | None, list[dict]]:
    context_length = int(metadata["context_length"])
    stats, global_mean, global_std = compute_train_split_scaler(frame)

    samples = []
    rows = []
    industry_ids = []
    unknown_id = industry_to_id.get("Unknown", 0)

    for ticker, group in frame.groupby(TICKER_COLUMN, sort=False):
        group = group.sort_values(TIMESTAMP_COLUMN)
        if len(group) < context_length:
            continue
        context = group.tail(context_length).copy()
        mean, std = stats.get(str(ticker), (global_mean, global_std))
        context.loc[:, TARGET_COLUMNS] = (context[TARGET_COLUMNS].astype(float) - mean) / std
        samples.append(context[TARGET_COLUMNS].astype("float32").to_numpy())

        industry = ticker_industry.get(str(ticker).upper(), "Unknown")
        industry_ids.append(industry_to_id.get(industry, unknown_id))
        rows.append(
            {
                "ticker": str(ticker),
                "sector": str(group[SECTOR_COLUMN].dropna().iloc[-1]) if SECTOR_COLUMN in group else "Unknown",
                "industry": industry,
                "context_start": group.tail(context_length)[TIMESTAMP_COLUMN].iloc[0],
                "context_end": group.tail(context_length)[TIMESTAMP_COLUMN].iloc[-1],
                "last_close": float(group["Close"].iloc[-1]),
            }
        )

    if not samples:
        raise ValueError(f"No tickers had at least {context_length} rows.")

    past_values = torch.tensor(np.stack(samples), dtype=torch.float32)
    industry_tensor = torch.tensor(industry_ids, dtype=torch.long) if industry_to_id else None
    return past_values, industry_tensor, rows


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=axis, keepdims=True)


def build_recommendations(logits: np.ndarray, sample_rows: list[dict]) -> pd.DataFrame:
    probs = softmax(logits, axis=-1)
    pred_classes = logits.argmax(axis=-1)
    rows = []
    run_timestamp = pd.Timestamp.utcnow()

    for sample_idx, sample in enumerate(sample_rows):
        context_end = pd.Timestamp(sample["context_end"])
        forecast_dates = pd.bdate_range(context_end + pd.offsets.BDay(1), periods=logits.shape[1])
        for day_idx in range(logits.shape[1]):
            pred_class = int(pred_classes[sample_idx, day_idx])
            direction = CLASS_NAMES[pred_class]
            confidence = float(probs[sample_idx, day_idx, pred_class])
            rows.append(
                {
                    "run_timestamp": run_timestamp,
                    "ticker": sample["ticker"],
                    "sector": sample["sector"],
                    "industry": sample["industry"],
                    "context_start": sample["context_start"],
                    "context_end": sample["context_end"],
                    "forecast_day": day_idx + 1,
                    "forecast_date": forecast_dates[day_idx],
                    "predicted_class": pred_class,
                    "predicted_direction": direction,
                    "recommendation": RECOMMENDATIONS[direction],
                    "confidence": confidence,
                    "prob_down": float(probs[sample_idx, day_idx, 0]),
                    "prob_flat": float(probs[sample_idx, day_idx, 1]),
                    "prob_up": float(probs[sample_idx, day_idx, 2]),
                    "last_close": sample["last_close"],
                }
            )

    return pd.DataFrame(rows).sort_values(["forecast_day", "recommendation", "confidence"], ascending=[True, True, False])


def recommendation_records(frame: pd.DataFrame) -> list[dict]:
    records = frame.copy()
    date_columns = ["run_timestamp", "context_start", "context_end", "forecast_date"]
    for column in date_columns:
        records[column] = pd.to_datetime(records[column]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        if column in {"context_start", "context_end", "forecast_date"}:
            records[column] = records[column].str.slice(0, 10)
    records = records.replace({np.nan: None})
    return records.to_dict(orient="records")


def upsert_recommendations(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    try:
        from supabase import create_client
    except ImportError as exc:
        raise SystemExit("Install supabase first: pip install supabase") from exc

    client = create_client(args.supabase_url, args.supabase_key)
    rows = recommendation_records(frame)
    if not rows:
        print("No recommendation rows to upsert.")
        return

    print(f"Upserting {len(rows):,} recommendation row(s) to {args.recommendations_table}")
    for start in range(0, len(rows), args.upsert_batch_size):
        batch = rows[start : start + args.upsert_batch_size]
        client.table(args.recommendations_table).upsert(
            batch,
            on_conflict="ticker,context_end,forecast_day",
        ).execute()
        print(f"  Upserted rows {start + 1:,}-{start + len(batch):,}")


def main() -> None:
    args = parse_args()
    load_ml_dependencies()
    device = select_device(args.device)
    print(f"Loading model from {args.model_dir}")
    model, metadata = load_model(args.model_dir, device)

    print("Loading historic data")
    history = load_history(args)
    tickers = sorted(history[TICKER_COLUMN].dropna().astype(str).str.upper().unique())
    print(f"Loaded {len(history):,} rows for {len(tickers):,} ticker(s)")

    ticker_industry, industry_to_id = load_industry_inputs(
        args,
        tickers,
        expected_count=int(metadata.get("num_industries", 0)),
    )
    past_values, industry_ids, sample_rows = make_latest_batch(history, metadata, ticker_industry, industry_to_id)
    past_values = past_values.to(device)
    if industry_ids is not None:
        industry_ids = industry_ids.to(device)

    print(f"Running inference for {len(sample_rows):,} ticker(s) on {device}")
    with torch.no_grad():
        outputs = model(past_values=past_values, industry_id=industry_ids)
    logits = outputs.logits.detach().cpu().numpy()

    recommendations = build_recommendations(logits, sample_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(args.output, index=False)
    if args.write_supabase:
        upsert_recommendations(recommendations, args)

    day1 = recommendations[recommendations["forecast_day"] == 1].copy()
    print(f"Wrote {len(recommendations):,} rows to {args.output}")
    print("\nTop day-1 BUY candidates:")
    print(
        day1[day1["recommendation"] == "BUY"]
        .sort_values("confidence", ascending=False)
        .head(10)[["ticker", "sector", "predicted_direction", "confidence", "prob_up", "prob_flat", "prob_down"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
