#!/bin/bash
set -e

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: No virtual environment detected!"
    echo "Please activate your virtual environment first:"
    echo "  source venv/bin/activate"
    exit 1
fi

PACKAGE_PREFIX="mongodb_ai_widget_challenge-"

echo "Building JavaScript assets..."
cd mongodb_ai_widget_challenge
npm install
npm run build
cd ..

echo "Building Python package..."
# Use pip from the virtual environment directly
"$VIRTUAL_ENV/bin/pip" install --upgrade build

echo "Removing stale build artifacts for ${PACKAGE_PREFIX}..."
mkdir -p dist
rm -f "dist/${PACKAGE_PREFIX}"*.whl "dist/${PACKAGE_PREFIX}"*.tar.gz

# Build the package using venv's python
"$VIRTUAL_ENV/bin/python" -m build

echo "Build complete! Artifacts are in the dist/ directory."
