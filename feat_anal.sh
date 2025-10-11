#!/usr/bin/env bash
set -e

WIN="${1:-rect}"   # 默认 hamming

python3 feature_analysis.py \
  --window "$WIN" \
  --bins 30 \
  --dpi 300 \
  --plots_root "/home/wdai/dw/DSP_exp1/plots"
