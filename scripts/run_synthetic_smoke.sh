#!/usr/bin/env bash
# Offline smoke run on SYNTHETIC (non-real) data. Use ONLY to verify the toolchain; never report these metrics.
set -euo pipefail
export PYTHONPATH=src
python -m pmip.cli synthetic --rows 6000
python -m pmip.cli eda
python -m pmip.cli train --fast
