#!/usr/bin/env bash
set -euo pipefail

echo "=== ArrowSpace MCP Setup ==="

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-genefold}"
if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
    uv venv "$UV_PROJECT_ENVIRONMENT" --python 3.13
fi

if git rev-parse --git-dir >/dev/null 2>&1; then
    echo "Updating submodules (arrowspace-skills)..."
    git submodule update --init --recursive
fi

uv sync --extra spectral "$@"

echo "Done. Try: UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT uv run arrowspace-mcp --help"
