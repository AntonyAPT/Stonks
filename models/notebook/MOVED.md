# This folder has been refactored

The contents of `models/notebook/` have been reorganised:

| Old location | New location |
|---|---|
| `models/notebook/labeling.py` | `models/patchtst_lib/labeling.py` |
| `models/notebook/classification_head.py` | `models/patchtst_lib/classification_head.py` |
| `models/notebook/dataset_utils.py` | `models/patchtst_lib/technical/dataset.py` |
| `models/notebook/backtest.py` | `models/patchtst_lib/technical/backtest.py` |
| `models/notebook/patchtst-new-branch-test-sector.ipynb` | `models/notebooks/technical/patchtst-stock-classifier.ipynb` |
| `models/notebook/patchtst-eval-backtest.ipynb` | `models/notebooks/technical/patchtst-eval-backtest.ipynb` |
| `models/notebook/ts_padding_utils.py` | **Deleted** (Granite baseline removed) |

Kaggle kernels should be re-pushed from the new notebook folders:
- Technical pipeline: `kaggle kernels push -p models/notebooks/technical`
- Fundamental pipeline: `kaggle kernels push -p models/notebooks/fundamental`
