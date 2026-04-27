#!/usr/bin/env zsh
set -euo pipefail

# Creates a local virtualenv at .venv and installs dependencies.
# Usage: ./setup_dev.sh

HERE=$(cd "$(dirname "$0")" && pwd)
VENV_DIR="$HERE/.venv"

if [ -d "$VENV_DIR" ]; then
  echo "Using existing virtualenv at $VENV_DIR"
else
  echo "Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtualenv..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing requirements..."
pip install --upgrade pip setuptools wheel
pip install -r "$HERE/requirements.txt"

echo "Setup complete. To use the venv run: source $VENV_DIR/bin/activate"
echo "You can then run: python -u src/pvt_analyzer.py"
