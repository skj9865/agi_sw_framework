#!/bin/bash
# Unified SW Framework - one-step environment setup
# Usage: bash install.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
#PYTHON= #수동으로 python 경로 지정하려면 여기에 설정 (예: PYTHON=/path/to/python3.10)

# Find Python 3.10 automatically
PYTHON="${PYTHON:-$(command -v python3.10 || command -v python3 || command -v python)}"
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
echo "Using Python $PY_VERSION at: $PYTHON"
if [[ "$PY_VERSION" != "3.10" ]]; then
    echo "WARNING: Python 3.10 recommended (found $PY_VERSION). Set PYTHON env var to override."
    echo "  e.g.: PYTHON=/path/to/python3.10 bash install.sh"
fi

# 1. Create venv if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating venv..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "[1/4] venv already exists, skipping"
fi

# Detect OS for venv paths
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    PIP="$VENV_DIR/Scripts/pip.exe"
    VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
else
    PIP="$VENV_DIR/bin/pip"
    VENV_PYTHON="$VENV_DIR/bin/python"
fi

# 2. Upgrade pip
echo "[2/4] Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet

# 3. Install all dependencies
echo "[3/4] Installing dependencies..."
"$PIP" install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 --quiet
"$PIP" install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html --quiet
"$PIP" install -r "$SCRIPT_DIR/requirements.txt" --quiet

# 4. Install tbp.monty (editable, no-deps to avoid conflicts)
echo "[4/4] Installing tbp.monty..."
"$PIP" install --no-deps -e "$SCRIPT_DIR/algorithms/monty/tbp.monty/"

echo ""
echo "Done! Activate with: .venv/Scripts/activate"
echo "Then run: python run.py --list"
