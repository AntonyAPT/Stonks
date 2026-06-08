# PatchTST Stock Classifier Notebook (Technical — OHLCV)

This folder (`models/notebooks/technical/`) contains the training and evaluation
notebooks for the **technical** PatchTST pipeline, which operates on daily OHLCV data.
Shared library code lives in `models/patchtst_lib/`.
The fundamentals pipeline lives in `models/notebooks/fundamental/`.

The notebook predicts one class for each of the next five trading days:

- `0`: significantly lower close
- `1`: roughly flat close
- `2`: significantly higher close

The first working label rule is a fixed percentage threshold. You can later swap
it for rolling-volatility or ATR thresholds by changing `LABEL_RULE` and related
hyperparameters in the notebook config cell.

**Primary workflow: Kaggle** (free GPU, no local environment needed).
Local execution (WSL2 / macOS) is supported as a fallback — see the [Local Setup](#local-setup-fallback) section.

---

## Quick-start: Kaggle

### One-time setup (done once per team, not per member)

**1. Upload the shared raw-data dataset**

From the repo root, run:

```bash
pip install kaggle  
# if not already installed (doesn't matter in what directory but ../notebook is fine)
# should be the latest CLI, look at 'kaggle_cli_setup.md' for more info
# create models/data_raw/dataset-metadata.json first (see below)
# kaggle datasets create -p models/data_raw (this for the initial creation of the dataset which has been done).
```

`models/data_raw/dataset-metadata.json` should contain:

```json
{
  "title": "SP500 Daily Raw",
  "id": "<TEAM_KAGGLE_USER>/sp500-daily-raw",
  "licenses": [{ "name": "CC0-1.0" }]
}
```

When raw CSVs change, version the dataset:

```bash
kaggle datasets version -p models/data_raw -m "describe the update"
# run this command when you add new data locally (or just upload it to kaggle directly)
```

**2. Configure your Kaggle CLI Authentication (Allows kaggle commands [dataset, kernel] to run)**  

Download from *kaggle.com → Settings → API → Create New Token (or Create Legacy API call*).

Note: Sanity Checks provided by Kaggle during this process.

**3. (Optional) Add a GitHub PAT secret to Kaggle**

If the repository is **private**, go to *kaggle.com → Your Profile → Settings → Secrets* and add a secret named `GITHUB_PAT` containing a fine-grained personal access token with `Contents: read` permission. The bootstrap cell in the notebook includes commented-out code to consume it.

---

## Per-member sandbox workflow

Each team member runs their own private Kaggle kernel tied to a git feature branch. GitHub remains the single source of truth — Kaggle kernels are ephemeral GPU environments, not a code store.

### 1. Create a feature branch

```bash
git checkout -b feature/<short-name>
git push -u origin feature/<short-name>
```

### 2. Configure your kernel metadata

Copy `kernel-metadata.json` and set your own slug:

```bash
# The file lives at models/notebook/kernel-metadata.json
# Edit it: set "id" to "<your-kaggle-user>/patchtst-<short-name> 
# Note: 'patchtst-<short-name>' and 'title' in this file must match else 'kernal push' fails
# This is giving your notebook execution environment (i.e. kaggle kernel) a unique slug url/identifier so you are also free to change what comes after '/'. "
```

### 3. Point the bootstrap cell at your branch

Open `patchtst_stock_classifier.ipynb` and change one line in the bootstrap cell:

```python
REPO_BRANCH = 'feature/<short-name>'   # was 'main'
```

Then commit and push that change:

```bash
git add models/notebook/patchtst_stock_classifier.ipynb models/notebook/kernel-metadata.json
git commit -m "chore: configure Kaggle sandbox for feature/<short-name>"
git push
```

# Kaggle Training Workflow

### 4. Push the kernel to Kaggle

```bash
kaggle kernels push -p models/notebooks/technical --accelerator NvidiaTeslaT4
# accelerator flag determines what gpu to use for model training
# must have latest Kaggle CLI installed. Look at 'kaggle_cli_setup.md' for more info and troubleshooting 
# Note: To use gpu's you need to verify your account with your phone number
```

This creates (or updates) your private kernel on Kaggle. On first push it may take a minute to provision.

### 5. Pull Artifacts After the Run Finishes

Run the pull script from the `models/notebook` directory (May need to be edited if you alter the notebook/kernel name):

```bash
bash pull_results.sh   # from models/notebooks/technical/
```

This pulls the training artifacts (checkpoints and saved models) to your local machine.

---

### 6. Download the Executed Notebook

Go to `kaggle.com/<your-user>/patchtst-<short-name>`, open the finished version, and click **File → Download Notebook** to get the notebook with cell outputs. Replace `models/notebook/patchtst_stock_classifier.ipynb` with the downloaded file.

---

### 7. Commit Results to GitHub

```bash
git add .
git commit -m "training run: <short description>"
git push
```

The shared **team canonical kernel** (`<team-account>/patchtst-stock-classifier`) always tracks `main` and is updated with `kaggle kernels push` after a PR merges.

---

## Architecture overview

```
GitHub (source of truth)
  main branch ──────────────────────────────► team/patchtst-stock-classifier (Kaggle)
  feature/* branches ──────────────────────► <member>/patchtst-<branch>   (Kaggle)

Kaggle Dataset: <team>/sp500-daily-raw
  attached as input to all kernels above

Flow:
  local edit → git push → kaggle kernels push → GPU run → kaggle kernels pull → git push → PR
```

---

## Local Setup (fallback)

> Local execution works but requires a manual environment and a local GPU or CPU-only training (slow). Prefer Kaggle for actual training runs.

### WSL2 (Windows)

**1. Install WSL2 and Ubuntu** — from PowerShell (Administrator):

```powershell
wsl --install -d Ubuntu-22.04
```

**2. Create a virtual environment:**

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
cd ../SeniorProject/models/notebook
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

> Tip: if file I/O feels slow, copy the project into the WSL home directory (`~/SeniorProject`) instead of working from `/mnt/c/...`.

**3. Install PyTorch with CUDA (RTX 3050 Ti / cu126):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

NVIDIA's Windows GPU driver is automatically shared with WSL2 — no separate Linux driver needed.

**4. Install remaining dependencies:**

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name patchtst-stock --display-name "patchtst-stock (WSL)"
```

**5. Select the kernel in Cursor:** choose **"patchtst-stock (WSL)"** from the kernel picker.

---

### macOS

```bash
cd models/notebook
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio   # MPS on Apple Silicon automatically
pip install -r requirements.txt
python -m ipykernel install --user --name patchtst-stock --display-name "patchtst-stock"
```

---

## Device Notes

The notebook selects the best available backend automatically:

```python
DEVICE_OVERRIDE = None   # None | 'cuda' | 'mps' | 'cpu'
DTYPE_OVERRIDE = None    # None | 'bfloat16' | 'float16' | 'float32'
```


| Setup                  | Auto-selected device | Auto-selected dtype               |
| ---------------------- | -------------------- | --------------------------------- |
| Kaggle GPU (T4 / P100) | `cuda`               | `bfloat16` or `float16`           |
| WSL2 + NVIDIA GPU      | `cuda`               | `bfloat16` (Ampere+) or `float16` |
| Apple Silicon Mac      | `mps`                | `float32`                         |
| CPU only               | `cpu`                | `float32`                         |


If you hit GPU out-of-memory errors, reduce `BATCH_SIZE` from `64` to `32` or `16` in the config cell.

---

## TensorBoard

Training logs are written to `models/notebooks/technical/checkpoint/patchtst_cls/` (local) or `/kaggle/working/checkpoint/patchtst_cls/` (Kaggle).

**Local:**

```bash
tensorboard --logdir ./checkpoint/patchtst_cls/
```

**Kaggle:** download the `checkpoint/` folder from the kernel output panel, then run TensorBoard locally against it.

---

## Local Inference

After pulling Kaggle outputs, generate a local recommendation CSV with:

```bash
cd models/notebook
python local_inference.py --tickers AAPL MSFT NVDA
```

For the full universe:

```bash
python local_inference.py --sp500-tickers
```

By default this reads `save_dir/patchtst_cls_sector_all_11_sectors`, fetches
historic rows from Supabase, and writes `save_dir/latest_recommendations.csv`.
If `ticker_industry.json` is missing, the script can still run by using the
model's `Unknown` industry embedding, but predictions are more faithful if you
provide or build the ticker-to-industry map.

To store the generated rows for the website:

1. Run `docs/model_recommendations.sql` and `docs/fundamental_recommendations.sql` once in the Supabase SQL editor.
2. Set a write-capable key locally, preferably the Supabase service role key:

```bash
export SUPABASE_RECOMMENDATIONS_KEY="<service-role-key>"
```

1. Run inference with Supabase upsert enabled:

```bash
python local_inference.py --tickers AAPL MSFT NVDA --write-supabase
```

For the daily automated workflow, set these GitHub repository secrets:

```text
KAGGLE_USERNAME
KAGGLE_KEY
SUPABASE_SERVICE_ROLE_KEY
SUPABASE_URL
```

Then run **Daily Model Recommendations** manually from the GitHub Actions tab
once to confirm it can pull the Kaggle model artifacts, update `historic_data`,
and upsert `model_recommendations`. After that it runs daily on the cron
schedule in `.github/workflows/daily-model-recommendations.yml`.

---

## Known Issues / Gotchas

### `LazyLinear` + HuggingFace Trainer ≥ 4.56

`MultiDayClassificationHead` in `classification_head.py` uses `nn.LazyLinear`
so the wrapper stays robust to PatchTST output-shape variations across
`transformers` versions. Starting in `transformers` 4.56, `Trainer.train()`
calls `get_model_param_count(...)` (which invokes `.numel()` on every
parameter) **before** running the first forward pass, which raises:

```
ValueError: Attempted to use an uninitialized parameter in <method 'numel' ...>.
This error happens when you are using a `LazyModule` ...
```

**Fix already applied:** `PatchTSTClassifier.__init__` runs a single dummy
`forward()` at the end of construction (see `_materialize_lazy_params`) so the
lazy weights are concrete by the time any caller (Trainer, manual loops,
`save_pretrained`, etc.) touches them. No action needed at the notebook level.

### torch / torchvision / torchaudio version alignment (local only)

When installing locally, pin the three PyTorch packages together to avoid ABI
mismatches. See `requirements.txt` for notes. On Kaggle this is not an issue —
the preinstalled stack is already consistent.