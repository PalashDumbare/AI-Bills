#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="/tmp/ai-bills-uvicorn.pid"
PORT=8000

stop_existing_server() {
  local found=0
  # 1) pid file
  if [ -f "$PID_FILE" ]; then
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping previous server (pid $pid)..."
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
      found=1
    fi
    rm -f "$PID_FILE" 2>/dev/null || true
  fi
  # 2) lsof on port (macOS/Linux)
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      echo "Stopping server on :$PORT (pids: $pids)..."
      echo "$pids" | xargs kill 2>/dev/null || true
      sleep 1
      echo "$pids" | xargs kill -9 2>/dev/null || true
      found=1
    fi
  fi
  # 3) pgrep fallback
  if pgrep -f "uvicorn app.main:app" >/dev/null 2>&1; then
    echo "Stopping uvicorn processes..."
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    sleep 1
    pkill -9 -f "uvicorn app.main:app" 2>/dev/null || true
    found=1
  fi
  if [ "$found" -eq 1 ]; then
    sleep 1
    echo "Previous server stopped."
  fi
}

# If script is run again while server is running, restart it
# Also supports: ./start.sh stop | restart | start
case "${1:-}" in
  stop)
    stop_existing_server
    echo "Stopped."
    exit 0
    ;;
  restart)
    stop_existing_server
    ;;
  "")
    # default: if server already running, stop it first (restart behavior)
    if [ -f "$PID_FILE" ] || (command -v lsof >/dev/null 2>&1 && lsof -ti :"$PORT" >/dev/null 2>&1) || pgrep -f "uvicorn app.main:app" >/dev/null 2>&1; then
      echo "==> Server already running — restarting..."
      stop_existing_server
    fi
    ;;
  *)
    echo "Usage: $0 [stop|restart]"
    exit 1
    ;;
esac

VENV="$ROOT/Bills/.venv"
# fallback venv locations
if [ ! -f "$VENV/bin/uvicorn" ]; then
  if [ -f "$ROOT/.venv/bin/uvicorn" ]; then VENV="$ROOT/.venv"
  elif [ -f "$ROOT/backend/.venv/bin/uvicorn" ]; then VENV="$ROOT/backend/.venv"
  fi
fi
PYTHON="$VENV/bin/python"
ALEMBIC="$VENV/bin/alembic"
if [ ! -f "$PYTHON" ]; then PYTHON="python3"; fi
if [ ! -f "$ALEMBIC" ]; then ALEMBIC="alembic"; fi

echo "==> [1/5] Postgres (homebrew postgresql@16)..."
if ! pg_isready -q 2>/dev/null; then
  brew services start postgresql@16 2>/dev/null || brew services start postgresql 2>/dev/null || true
  echo "Waiting for Postgres..."
  for i in {1..15}; do pg_isready -q && break; sleep 1; done
fi
pg_isready -q || { echo "Postgres failed to start. Try: brew services start postgresql@16"; exit 1; }
# create DB if missing (uses DATABASE_URL from backend/.env or default)
createdb ai_bills 2>/dev/null || true
psql -lqt 2>/dev/null | cut -d \| -f 1 | grep -qw ai_bills || { echo "DB ai_bills missing — creating via createdb"; createdb ai_bills || true; }

echo "==> [2/5] Ollama (llama3.2:3b)..."
if ! pgrep -x ollama >/dev/null 2>&1; then
  echo "Starting ollama serve..."
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 2
fi
# pull model if missing
if ! ollama list 2>/dev/null | grep -q "llama3.2"; then
  echo "Pulling llama3.2:3b (this takes a few minutes first time)..."
  ollama pull llama3.2:3b
fi

echo "==> [3/5] Qdrant (vector DB) on :6333..."
mkdir -p "$ROOT/qdrant_storage"
# check if already responding
if ! curl -sf http://localhost:6333/ >/dev/null 2>&1; then
  # ensure docker daemon
  if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon not running — starting Docker Desktop..."
    open -a Docker 2>/dev/null || true
    echo "Waiting for Docker (max 60s)..."
    for i in {1..30}; do docker info >/dev/null 2>&1 && break; sleep 2; done
  fi
  if docker info >/dev/null 2>&1; then
    # reuse existing container if present
    if docker ps -a --format '{{.Names}}' | grep -qx qdrant; then
      echo "Starting existing qdrant container..."
      docker start qdrant >/dev/null 2>&1 || true
    else
      echo "Creating qdrant container..."
      docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "$ROOT/qdrant_storage:/qdrant/storage" qdrant/qdrant >/dev/null
    fi
    echo "Waiting for Qdrant..."
    for i in {1..15}; do curl -sf http://localhost:6333/ >/dev/null && break; sleep 1; done
  else
    echo "WARNING: Docker still not ready. Qdrant will not start."
    echo "Start Docker manually and run: docker run -d --name qdrant -p 6333:6333 -v \"$ROOT/qdrant_storage:/qdrant/storage\" qdrant/qdrant"
  fi
fi
curl -sf http://localhost:6333/ >/dev/null && echo "Qdrant OK: http://localhost:6333/dashboard" || echo "Qdrant not reachable — backend RAG will fail until Qdrant is up"

echo "==> [4/5] Python deps + DB migrations..."
if [ -f "$ROOT/requirements.txt" ]; then
  "$PYTHON" -c "import fastapi" 2>/dev/null || "$VENV/bin/pip" install -r "$ROOT/requirements.txt"
fi
mkdir -p "$ROOT/backend/storage/documents" "$ROOT/backend/app/extracted"
# alembic needs to run from backend dir (where alembic.ini lives)
(cd "$ROOT/backend" && "$ALEMBIC" upgrade head 2>&1 | tail -n 20)

echo "==> [5/5] Starting FastAPI (http://localhost:8000/docs)..."
echo "Logs: uvicorn + postgres + ollama (/tmp/ollama.log) + qdrant"
# pid file + cleanup on exit (covers second-run restart)
echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE" 2>/dev/null || true' EXIT INT TERM
# Use venv python to ensure correct deps
exec "$PYTHON" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir "$ROOT/backend"
