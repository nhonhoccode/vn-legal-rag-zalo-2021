#!/usr/bin/env bash
# ============================================================
# VN Legal RAG — Auto installer
# ------------------------------------------------------------
# Kiểm tra prerequisites, sinh .env từ template, generate JWT_SECRET,
# hỏi các API key tối thiểu, build images và khởi động stack.
#
# Usage:
#   bash scripts/setup.sh             # interactive
#   bash scripts/setup.sh --yes       # non-interactive (cần env vars sẵn)
# ============================================================
set -euo pipefail

NON_INTERACTIVE=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes) NON_INTERACTIVE=1 ;;
    -h|--help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
  esac
done

# ----- helpers -----
log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Thiếu lệnh '$1'. ${2:-Vui lòng cài đặt rồi chạy lại.}"
    exit 1
  fi
}

# ----- 0. cd repo root -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
log "Repo root: $REPO_ROOT"

# ----- 1. Prerequisites -----
log "Kiểm tra prerequisites..."
require_cmd docker "Cài Docker theo https://docs.docker.com/engine/install/"
if ! docker compose version >/dev/null 2>&1; then
  err "Docker Compose v2 không khả dụng. Vui lòng cập nhật Docker Engine."
  exit 1
fi
require_cmd openssl "openssl cần có để generate JWT_SECRET."
ok "Docker $(docker --version | awk '{print $3}' | tr -d ,), Compose v2 ready."

# ----- 2. Env files -----
init_env() {
  local example="$1" target="$2"
  if [ -f "$target" ]; then
    warn "$target đã tồn tại, giữ nguyên."
    return
  fi
  if [ ! -f "$example" ]; then
    err "Thiếu template $example."
    exit 1
  fi
  cp "$example" "$target"
  ok "Đã tạo $target từ $example."
}

log "Khởi tạo các file .env..."
init_env ".env.example" ".env"
init_env "backend/.env.example" "backend/.env"
if [ -f "frontend/.env.example" ]; then
  init_env "frontend/.env.example" "frontend/.env.local"
fi

# ----- 3. JWT_SECRET -----
if grep -q '^JWT_SECRET=change-me' backend/.env || \
   grep -q '^JWT_SECRET=$' backend/.env; then
  NEW_SECRET="$(openssl rand -hex 32)"
  # in-place edit (POSIX-safe via temp file)
  awk -v s="$NEW_SECRET" '
    /^JWT_SECRET=/ { print "JWT_SECRET=" s; next }
    { print }
  ' backend/.env > backend/.env.tmp && mv backend/.env.tmp backend/.env
  ok "Đã generate JWT_SECRET ngẫu nhiên 32 bytes."
else
  ok "JWT_SECRET đã được set sẵn, giữ nguyên."
fi

# ----- 4. API keys -----
prompt_api_key() {
  local key_name="$1" prompt_text="$2" current
  current="$(grep -E "^${key_name}=" backend/.env | head -1 | cut -d= -f2-)"
  if [ -n "$current" ]; then
    ok "$key_name đã được set."
    return
  fi
  if [ "$NON_INTERACTIVE" -eq 1 ]; then
    warn "$key_name trống. Set thủ công trong backend/.env trước khi chạy stack."
    return
  fi
  printf '\n%s\n' "$prompt_text"
  read -r -p "  $key_name = " value || true
  if [ -n "$value" ]; then
    awk -v k="$key_name" -v v="$value" '
      $0 ~ "^"k"=" { print k "=" v; next }
      { print }
    ' backend/.env > backend/.env.tmp && mv backend/.env.tmp backend/.env
    ok "Đã set $key_name."
  else
    warn "Bỏ qua $key_name, set sau trong backend/.env."
  fi
}

if [ "$NON_INTERACTIVE" -eq 0 ]; then
  log "Cấu hình API keys (nhấn Enter để bỏ qua)..."
  prompt_api_key "OPENROUTER_API_KEY" "OpenRouter API key (sk-or-v1-...): https://openrouter.ai/keys"
  prompt_api_key "GEMINI_API_KEY"     "Google AI Studio API key (tuỳ chọn): https://aistudio.google.com/app/apikey"
fi

# ----- 5. Data check -----
if [ ! -d "data/chroma_db" ] || [ -z "$(ls -A data/chroma_db 2>/dev/null || true)" ]; then
  warn "data/chroma_db rỗng. Backend sẽ chạy được nhưng retrieval không có index."
  warn "Tải pre-built từ GitHub Releases hoặc build qua backend/scripts/04_embed.py."
else
  ok "ChromaDB index sẵn sàng tại data/chroma_db/"
fi

if [ ! -f "data/raw/zalo_legal/qa.jsonl" ]; then
  warn "Thiếu dữ liệu Zalo AI 2021. Tải từ https://challenge.zalo.ai/ vào data/raw/zalo_legal/"
fi

# ----- 6. Build & up -----
log "Build và khởi động Docker Compose stack..."
docker compose pull --ignore-pull-failures
docker compose up -d --build

ok "Stack đã khởi động. Kiểm tra trạng thái:"
docker compose ps

cat <<'EOF'

============================================================
Setup hoàn tất.

  Frontend     http://localhost:3000
  Backend API  http://localhost:8000/docs

Lệnh thường dùng:
  docker compose ps                     # trạng thái container
  docker compose logs -f api            # log backend realtime
  docker compose down                   # dừng toàn bộ
  docker compose up -d --build          # rebuild + restart

Nếu bạn chưa có ChromaDB index, tải pre-built từ GitHub Releases
hoặc chạy backend/scripts/04_embed.py để build từ scratch.
============================================================
EOF
