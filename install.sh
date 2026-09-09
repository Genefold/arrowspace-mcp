#!/usr/bin/env bash
set -euo pipefail

echo "=== ArrowSpace MCP Setup ==="

if git rev-parse --git-dir >/dev/null 2>&1; then
    echo "Updating submodules (arrowspace-skills)..."
    git submodule update --init --recursive
fi

if ! python -c "import arrowspace" 2>/dev/null; then
    echo "Installing arrowspace..."
    python -m pip install arrowspace
fi

python -m pip install -e .

echo "Done. Try: uv run arrowspace-mcp --help"
