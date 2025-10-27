#!/usr/bin/env bash
set -euo pipefail
TARGET="${1:-$HOME/.venvs/sail-tf1b}"

python3 -m venv "$TARGET"
source "$TARGET/bin/activate"
python -m pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements.txt"

python -V
pip list | wc -l
echo "Venv ready at: $TARGET"
