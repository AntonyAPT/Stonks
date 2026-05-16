#!/bin/bash
KERNEL=${1:-"kingz101/patchtst-new-branch-test-sector"} # matches 'id' in kernel-metadata.json

echo "Pulling training artifacts and predictions..."

# assuming you run from models/notebook
kaggle kernels output $KERNEL \
  -p . \
  --file-pattern "(^|/)(checkpoint|save_dir)/.*" \
  -o

echo "Pulling pred_df_test.csv for local backtest experiments..."
kaggle kernels output $KERNEL \
  -p . \
  --file-pattern "pred_df_test.csv" \
  -o

echo "Done. Run backtest_local.py to compare weighting strategies without the model."