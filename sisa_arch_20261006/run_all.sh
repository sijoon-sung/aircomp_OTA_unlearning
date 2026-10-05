#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
command -v "$PY" > /dev/null 2>&1 || PY=python
"$PY" -c "import torch, torchvision, numpy" || { echo "[!] torch/torchvision/numpy not found. See README_KO.md section 2."; exit 1; }
PYTHONIOENCODING=utf-8 "$PY" run_all.py "$@"
echo "Done. Open runs/<latest folder>/SUMMARY_KO.md"
