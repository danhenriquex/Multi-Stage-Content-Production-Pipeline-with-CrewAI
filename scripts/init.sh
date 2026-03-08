#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# init.sh — First-time project bootstrap
# Installs uv, syncs dependencies, sets up pre-commit, and creates .env
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
error()   { echo -e "${RED}✗${NC} $*"; exit 1; }
section() { echo -e "\n${BOLD}$*${NC}"; }

# ── 1. Install uv if missing ──────────────────────────────────────────────────
section "=== Content Pipeline – Init ==="

if ! command -v uv &> /dev/null; then
    warn "'uv' not found. Installing via official script..."
    if command -v curl &> /dev/null; then
        curl -Ls https://astral.sh/uv/install.sh | bash
    elif command -v wget &> /dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | bash
    else
        error "Neither curl nor wget found. Install one and retry."
    fi

    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

    command -v uv &> /dev/null || error "uv installation failed. Install manually: https://docs.astral.sh/uv/"
    info "uv installed: $(uv --version)"
else
    info "uv already installed: $(uv --version)"
fi

# ── 2. Sync Python dependencies ───────────────────────────────────────────────
section "Syncing Python dependencies..."
uv venv --python 3.11
uv pip install -e ".[dev]"
info "Dependencies synced"

# ── 3. Pre-commit hooks ───────────────────────────────────────────────────────
section "Installing pre-commit hooks..."
if uv run -- pre-commit install && uv run -- pre-commit install --hook-type pre-push; then
    info "pre-commit hooks installed (commit + push)"
else
    warn "pre-commit install failed — skipping (non-fatal)"
fi

# ── 4. Create .env from example ──────────────────────────────────────────────
section "Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    info "Created .env from .env.example"
    echo ""
    echo -e "  ${YELLOW}ACTION REQUIRED:${NC} Edit .env and set your API keys:"
    echo "    OPENAI_API_KEY=sk-..."
    echo "    POSTGRES_PASSWORD=your-password"
    echo ""
    echo "  Get your OpenAI key at: https://platform.openai.com/account/api-keys"
else
    info ".env already exists — skipping (delete it to reset)"
fi

# ── 5. Check Docker ───────────────────────────────────────────────────────────
section "Checking Docker..."
if docker info &> /dev/null; then
    info "Docker is running"
else
    warn "Docker does not appear to be running"
    warn "Start Docker Desktop (or dockerd) before running 'make up'"
fi

# ── 6. Create dagster_home if missing ────────────────────────────────────────
section "Checking Dagster home..."
if [ ! -f dagster_home/dagster.yaml ]; then
    mkdir -p dagster_home
    cat > dagster_home/dagster.yaml << 'YAML'
storage:
  sqlite:
    base_dir: /dagster_home
YAML
    info "Created dagster_home/dagster.yaml"
else
    info "dagster_home already exists — skipping"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Setup complete! Next steps:${NC}"
echo "  1. Edit .env with your API keys  →  vim .env"
echo "  2. Build + start all services    →  make build && make up"
echo "  3. Verify health                 →  make health"
echo "  4. Trigger a test campaign       →  make campaign"
echo "  5. Watch live agent logs         →  make logs-api"
echo "  6. View ML runs                  →  http://localhost:5050"
echo "  7. View pipeline graph           →  http://localhost:3001"
