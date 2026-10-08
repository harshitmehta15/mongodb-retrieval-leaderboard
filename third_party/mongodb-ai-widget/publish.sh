#!/bin/bash
set -e

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo "Error: dist directory not found. Please run ./build.sh first."
    exit 1
fi

VERSION=$(python3 - <<'PY'
from pathlib import Path
import re

pyproject = Path("pyproject.toml").read_text()
match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
if not match:
    raise SystemExit("Could not determine version from pyproject.toml")
print(match.group(1))
PY
)

WHEEL="dist/mongodb_ai_widget_challenge-${VERSION}-py3-none-any.whl"
SDIST="dist/mongodb_ai_widget_challenge-${VERSION}.tar.gz"

if [ ! -f "$WHEEL" ] || [ ! -f "$SDIST" ]; then
    echo "Error: expected artifacts for version ${VERSION} were not found."
    echo "Expected:"
    echo "  $WHEEL"
    echo "  $SDIST"
    echo "Please run ./build.sh first."
    exit 1
fi

echo "Installing twine..."
python3 -m pip install --upgrade twine

echo "Uploading to PyPI..."
# By default this will prompt for username/password or API token
# To use an API token, set username to __token__ and password to your token
# You can also use TestPyPI by adding: --repository testpypi
python3 -m twine upload "$WHEEL" "$SDIST"
