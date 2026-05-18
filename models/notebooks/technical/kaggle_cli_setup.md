# Kaggle CLI Setup — Known Issues & Fixes

## Issue 1 — Kaggle CLI version too old

The `--accelerator` flag and other features require Kaggle CLI 2.1.0+. The default pip install may give you an outdated version (1.7.4.5).

**Fix:** Check your version first:

```bash
kaggle --version
```

If below 2.1.0, upgrade:

```bash
pip install --upgrade kaggle
```

---

## Issue 2 — Upgrade doesn't work due to Python 3.10

Kaggle CLI 2.1.0 requires Python 3.11+. If your system runs Python 3.10, pip will silently skip the upgrade.

**Fix:** Install Python 3.11 and pip for it:

```bash
curl https://bootstrap.pypa.io/get-pip.py | python3.11
python3.11 -m pip install --upgrade kaggle
```

Verify with:

```bash
kaggle --version  # should show 2.1.0
```

---

## Issue 3 — Accelerator not picked up from kernel-metadata.json

Setting `"enable_gpu": "true"` in `kernel-metadata.json` defaults to CPU. Setting `"accelerator": "NvidiaTeslaT4"` in the metadata file did not work reliably.

**Fix:** Pass the accelerator explicitly as a flag when pushing:

```bash
kaggle kernels push -p models/notebook --accelerator NvidiaTeslaT4
```

Use this command instead of the bare `kaggle kernels push` for all training runs until the metadata file issue is resolved.

---

## Note for WSL Users

Do not change your default `python3` to point to 3.11 as Ubuntu system tools depend on 3.10 and breaking that can cause hard to debug issues. The pip and kaggle upgrade steps above install everything under 3.11 without touching the system default.