#!/bin/bash
# Test KG Agent Pipeline - kills previous server, starts fresh, creates session, runs test
# Usage: ./test_kg_pipeline.sh [max_articles]
# Example: ./test_kg_pipeline.sh 100

set -e

PORT=8080
APP_NAME="kg_agent"
USER_ID="test_user"
MAX_ARTICLES=${1:-10}  # Default to 10 articles if not specified

echo "=== KG Agent Pipeline Test (max $MAX_ARTICLES articles) ==="
START=$(date +%s)

# 0. Kill any existing server
echo "[0] Killing any existing ADK server..."
pkill -f "adk api_server" 2>/dev/null || true
sleep 2

# 1. Start the server
echo "[1] Starting ADK server..."
cd /Users/dejandukic/dejan_dev/tetra/tetra_v1
uv run adk api_server kg_agent/ --port $PORT 2>&1 &
SERVER_PID=$!
echo "    Server PID: $SERVER_PID"

# Wait for server to be ready
echo "    Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:$PORT/list-apps > /dev/null 2>&1; then
        echo "    Server ready!"
        break
    fi
    sleep 1
done

# 2. Create a session
echo "[2] Creating session..."
SESSION_RESPONSE=$(curl -s http://localhost:$PORT/apps/$APP_NAME/users/$USER_ID/sessions \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"state": {}}')
SESSION_ID=$(echo $SESSION_RESPONSE | jq -r '.id')
echo "    Session ID: $SESSION_ID"

if [ "$SESSION_ID" == "null" ] || [ -z "$SESSION_ID" ]; then
    echo "ERROR: Failed to create session"
    echo "Response: $SESSION_RESPONSE"
    exit 1
fi

# 3. Run the pipeline test
echo "[3] Running pipeline test..."
echo "    Query: Build knowledge graph for orexin pathway (max $MAX_ARTICLES articles)"

RESPONSE=$(curl -s -X POST http://localhost:$PORT/run \
    -H "Content-Type: application/json" \
    -d "{
        \"app_name\": \"$APP_NAME\",
        \"user_id\": \"$USER_ID\",
        \"session_id\": \"$SESSION_ID\",
        \"new_message\": {
            \"role\": \"user\",
            \"parts\": [{
                \"text\": \"Build a knowledge graph for orexin signaling pathway with HCRTR1, HCRTR2, HCRT. Fetch up to $MAX_ARTICLES articles from PubMed.\"
            }]
        },
        \"streaming\": false
    }")

# 4. Parse and display results
echo ""
echo "=== Results ==="

# Get the build_knowledge_graph response
KG_RESULT=$(echo "$RESPONSE" | jq -r '.[] | select(.content.parts[0].functionResponse.name == "build_knowledge_graph") | .content.parts[0].functionResponse.response')

if [ -n "$KG_RESULT" ] && [ "$KG_RESULT" != "null" ]; then
    echo "Knowledge Graph Summary:"
    echo "$KG_RESULT" | jq '.'

    echo ""
    echo "=== Relationship Type Breakdown ==="
    echo "$KG_RESULT" | jq -r '.relationship_types | to_entries | sort_by(-.value) | .[] | "  \(.key): \(.value)"'
else
    echo "Warning: Could not find build_knowledge_graph response"
    echo "Checking for final text response..."
    FINAL_TEXT=$(echo "$RESPONSE" | jq -r '.[-1].content.parts[0].text // empty' 2>/dev/null)
    if [ -n "$FINAL_TEXT" ]; then
        echo "$FINAL_TEXT"
    else
        echo "Raw response (first 3000 chars):"
        echo "$RESPONSE" | head -c 3000
    fi
fi

END=$(date +%s)
ELAPSED=$((END - START))
echo ""
echo "=== Total Time: ${ELAPSED}s ==="

# Kill server after test
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
