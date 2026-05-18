#!/bin/bash
KERNEL=${1:-"kingz101/patchtst-fundamental-classifier"}

echo "Pulling fundamental training artifacts..."

# Run from models/notebooks/fundamental/
kaggle kernels output $KERNEL \
  -p . \
  --file-pattern "(^|/)(checkpoint|save_dir_fund)/.*" \
  -o

echo "Done."
