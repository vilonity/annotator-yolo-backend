#!/bin/bash
# Start all services: YOLO backend, Elysia backend, Frontend
# Usage: bash start-all.sh

set -e

YOLO_DIR="C:/Users/user/utils/annotator-yolo-backend"
ANNOTATOR_DIR="C:/Users/user/utils/annotator-tools"
PYTHON="$YOLO_DIR/.venv/Scripts/python.exe"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cleanup() {
    echo -e "\n${YELLOW}Stopping all services...${NC}"
    kill $YOLO_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $YOLO_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}All services stopped.${NC}"
}
trap cleanup EXIT

# --- 1. YOLO Backend (port 8002) ---
echo -e "${GREEN}[1/3] Starting YOLO backend on port 8002...${NC}"

# Generate SSL certs if missing
if [ ! -f "$YOLO_DIR/certs/key.pem" ]; then
    echo -e "${YELLOW}Generating SSL certificates...${NC}"
    cd "$YOLO_DIR"
    "$PYTHON" generate_certs.py
fi

cd "$YOLO_DIR"
"$PYTHON" main.py &
YOLO_PID=$!
echo -e "${GREEN}  YOLO backend started (PID: $YOLO_PID)${NC}"

# --- 2. Elysia Backend (port 8001) ---
echo -e "${GREEN}[2/3] Starting Elysia backend on port 8001...${NC}"
cd "$ANNOTATOR_DIR/backend-elysia"

if [ -z "$DB_URL" ]; then
    echo -e "${RED}ERROR: DB_URL environment variable is not set.${NC}"
    echo -e "${YELLOW}Set it with: export DB_URL=postgres://user:pass@localhost:5432/dbname${NC}"
    exit 1
fi

bun run src/index.ts &
BACKEND_PID=$!
echo -e "${GREEN}  Elysia backend started (PID: $BACKEND_PID)${NC}"

# --- 3. Frontend (port 5173) ---
echo -e "${GREEN}[3/3] Starting frontend on port 5173...${NC}"
cd "$ANNOTATOR_DIR"
npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}  Frontend started (PID: $FRONTEND_PID)${NC}"

echo ""
echo -e "${GREEN}=== All services running ===${NC}"
echo -e "  Frontend:      ${YELLOW}http://localhost:5173${NC}"
echo -e "  Elysia API:    ${YELLOW}http://localhost:8001${NC}"
echo -e "  YOLO backend:  ${YELLOW}https://127.0.0.1:8002${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

wait
