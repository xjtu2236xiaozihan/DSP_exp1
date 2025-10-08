#!/usr/bin/env bash
set -e


WIN="${1:-rect}"   # 默认用 hamming 窗口

python3 feature_extraction.py \
  --audio_dir "./dataset/audio_processed" \
  --output_dir "./dataset/features" \
  --frame_length 0.025 \
  --frame_shift 0.010 \
  --sample_rate 16000 \
  --window "$WIN" \
  --output_name "audio_features"
