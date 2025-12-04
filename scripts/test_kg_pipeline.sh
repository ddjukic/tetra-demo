#!/bin/bash
# Test KG Agent Pipeline - kills previous server, starts fresh, creates session, runs test

set -e

PORT=8080
APP_NAME="kg_agent"
USER_ID="test_user"

echo "=== KG Agent Pipeline Test ==="
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
echo "    Query: Get STRING network, search PubMed, get annotations, extract relationships, build graph"

RESPONSE=$(curl -s -X POST http://localhost:$PORT/run \
    -H "Content-Type: application/json" \
    -d "{
        \"app_name\": \"$APP_NAME\",
        \"user_id\": \"$USER_ID\",
        \"session_id\": \"$SESSION_ID\",
        \"new_message\": {
            \"role\": \"user\",
            \"parts\": [{
                \"text\": \"Get the STRING network for HCRTR1, HCRTR2, HCRT. Search PubMed for orexin signaling (max 10 results). Get entity annotations. Extract relationships. Build a knowledge graph.\"
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
    echo "$KG_RESULT" | jq '.summary'

    NODE_COUNT=$(echo "$KG_RESULT" | jq '.node_count')
    EDGE_COUNT=$(echo "$KG_RESULT" | jq '.edge_count')
    echo ""
    echo "Node Count: $NODE_COUNT"
    echo "Edge Count: $EDGE_COUNT"
else
    echo "Warning: Could not find build_knowledge_graph response"
    echo "Last response:"
    echo "$RESPONSE" | jq '.[-1].content.parts[0].text // .[-1]' 2>/dev/null || echo "$RESPONSE" | head -c 2000
fi

# Get entity annotation counts
ANNOT_RESULT=$(echo "$RESPONSE" | jq -r '.[] | select(.content.parts[0].functionResponse.name == "get_entity_annotations") | .content.parts[0].functionResponse.response')
if [ -n "$ANNOT_RESULT" ] && [ "$ANNOT_RESULT" != "null" ]; then
    echo ""
    echo "Entity Annotations:"
    echo "$ANNOT_RESULT" | jq '{pmids_annotated, entity_type_counts}'
fi

END=$(date +%s)
ELAPSED=$((END - START))
echo ""
echo "=== Total Time: ${ELAPSED}s ==="

# Optionally kill server after test (uncomment if desired)
# echo "Stopping server..."
# kill $SERVER_PID 2>/dev/null
