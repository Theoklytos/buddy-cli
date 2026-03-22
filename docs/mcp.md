# MCP Integration

Bud includes an MCP (Model Context Protocol) server that provides semantic search access to your conversation archive.

## Overview

The Bud MCP server is a stdio-based server built with FastMCP that provides read-only access to Bud's conversation archive through four tools:

- **bud_context** - Get session context and tool call history for Claude tracking
- **bud_recall** - Semantic search across your conversation archive
- **bud_orient** - Get structural snapshot and index statistics
- **bud_reflect** - Filter chunks by tag dimensions and values

All tools are read-only and log their activity to `~/mcp_logs/` for audit and replay purposes.

---

## Tools

### bud_context

Get the current MCP session context for Claude conversation tracking. Returns the session ID and recent tool call history so Claude can detect if this is a new session or continuation, and avoid duplicating tool calls.

**Input:**
```json
{
  "history_size": 10
}
```

**Output:**
```json
{
  "session_id": "mcp-abc123-def456",
  "is_new_session": false,
  "history": [
    {
      "tool_name": "bud_orient",
      "timestamp": "2026-03-22T10:30:00.123456Z",
      "parameters": {}
    },
    {
      "tool_name": "bud_recall",
      "timestamp": "2026-03-22T10:30:01.234567Z",
      "parameters": {"query": "auth system", "k": 5}
    }
  ],
  "total_calls_in_session": 2
}
```

**Use cases for Claude:**
- **Conversation Detection**: If `is_new_session: true`, Claude knows it's a fresh session
- **Duplicate Prevention**: Claude can check `history` to avoid repeating the same tool call
- **Context Persistence**: Claude can track which tools were already called in this session

---

### bud_recall

Search the archive using semantic similarity. This tool embeds your query and finds the most relevant chunks from your conversation archive.

**Input:**
```json
{
  "query": "your search query",
  "k": 5
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | Natural language query to search (max 2000 chars) |
| k | integer | No | Number of chunks to retrieve (1-20, default 5) |

**Output:**
```json
{
  "query": "your search query",
  "k": 5,
  "total_in_index": 1234,
  "results": [
    {
      "rank": 1,
      "chunk_id": "chunk-abc123",
      "score": 0.85,
      "chunk_type": "exchange",
      "tags": {"terrain": "conceptual"},
      "source_file": "conversation.json",
      "conversation_id": "conv-123",
      "turns": [1, 2, 3],
      "text": "..."
    }
  ]
}
```

---

### bud_orient

Get a structural snapshot of the conversation archive. This provides index statistics and a scatter sample of chunks for quick orientation.

**Input:** None

**Output:**
```json
{
  "index": {
    "total_chunks": 1234,
    "schema_version": 1,
    "dimensions": {"geometry": ["linear", "angular"]},
    "chunk_types": ["exchange", "breakthrough", "context"]
  },
  "sample": [
    {
      "chunk_id": "chunk-abc123",
      "chunk_type": "exchange",
      "tags": {"terrain": "conceptual"},
      "source_file": "conversation.json",
      "turns": [1, 2],
      "preview": "First 200 chars of the chunk..."
    }
  ]
}
```

The sample contains 12 randomly selected chunks (configurable in the code).

---

### bud_reflect

Retrieve chunks from the archive filtered by structural tag dimension and value. Use this to explore specific categories or themes in your archive.

**Input:**
```json
{
  "dimension": "terrain",
  "value": "emotional",
  "limit": 10
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| dimension | string | Yes | Tag dimension to filter (geometry, coherence, terrain, motifs, chunk_type, etc.) |
| value | string | Yes | Value within that dimension to match |
| limit | integer | No | Maximum results to return (1-50, default 10) |

**Output:**
```json
{
  "dimension": "terrain",
  "value": "emotional",
  "total_matched": 42,
  "returned": 10,
  "results": [
    {
      "chunk_id": "chunk-abc123",
      "chunk_type": "exchange",
      "tags": {"terrain": "emotional"},
      "source_file": "conversation.json",
      "text": "..."
    }
  ]
}
```

**Supported dimensions:**
- `chunk_type` - Filter by chunk type (exchange, breakthrough, context)
- `geometry` - Filter by structural geometry
- `coherence` - Filter by coherence markers
- `terrain` - Filter by terrain types
- `motifs` - Filter by motifs (supports multi-value arrays)

---

## Logging Architecture

All tool calls are logged to `~/mcp_logs/` with the following structure:

### Log Location
- **Directory**: `~/mcp_logs/` (separate from Bud's main data)
- **Format**: Per-session JSONL files
- **File naming**: `{session_id}.jsonl`
- **Retention**: Indefinite (user is responsible for cleanup)

### Log Entry Structure
```json
{
  "timestamp": "2026-03-22T10:30:00.123456Z",
  "session_id": "mcp-abc123-def456",
  "tool_name": "bud_recall",
  "parameters": {
    "query": "search text",
    "k": 5
  },
  "result": {
    "query": "search text",
    "k": 5,
    "total_in_index": 1234,
    "results": [...]
  },
  "duration_ms": 156.23
}
```

| Field | Type | Description |
|-------|------|-------------|
| timestamp | string | UTC ISO 8601 timestamp with timezone |
| session_id | string | Unique session identifier |
| tool_name | string | Name of the tool called |
| parameters | object | The parameters passed to the tool |
| result | object | The result returned by the tool |
| duration_ms | number | Execution time in milliseconds |

### Session Persistence
- Session IDs are persisted to `~/.mcp_logs/.current_session`
- Sessions can be resumed across CLI restarts
- Each unique session gets its own log file

---

## Session Context for Claude

The `bud_context` tool provides Claude with session tracking capabilities to maintain conversation state across MCP invocations.

### Session ID
- A unique identifier for the current MCP session (e.g., `mcp-abc123-def456`)
- Generated on first connection or loaded from `.current_session`
- Persists across CLI restarts but is unique per connection

### History
- The last N tool calls made in this session (default 10)
- Each entry contains: `tool_name`, `timestamp`, and `parameters`

### New Session Detection
- `is_new_session: true` indicates a fresh session (new ID)
- `is_new_session: false` indicates a resumed session (loaded from file)

### Use Cases

**1. Conversation Detection**
```json
{
  "session_id": "mcp-new-session-xyz",
  "is_new_session": true,
  "history": []
}
```
If `is_new_session` is true, Claude knows this is a fresh conversation.

**2. Duplicate Prevention**
Before making a tool call, Claude can check the history:
```json
{
  "session_id": "mcp-existing-session",
  "is_new_session": false,
  "history": [
    {"tool_name": "bud_recall", "parameters": {"query": "auth", "k": 5}}
  ]
}
```
If the same tool call with same parameters is already in history, Claude can skip it.

**3. Context Merging**
Claude can aggregate information from multiple tool calls:
```javascript
// Pseudo-code for Claude integration
const recentCalls = context.history.slice(-5);
// Use these to build a coherent response
```

---

## Configuration

### Claude Desktop (`claude_desktop_config.json`)

Add the MCP server to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "bud": {
      "command": "python",
      "args": ["/path/to/bud/bud_mcp.py"]
    }
  }
}
```

After updating, restart Claude Desktop to load the new server.

### CLI

Start the MCP server using the `bud mcp` command:

```bash
# Start MCP server with default session
bud mcp

# Start with custom session ID
bud mcp --session-id my-custom-session
```

**Options:**
- `--session-id` - Custom session ID for logging (optional)

### Manual Invocation

Run the server directly with Python:

```bash
# Using the module
python -m bud.bud_mcp

# Using the script
python /path/to/bud/bud_mcp.py
```

---

## Requirements

- **Python**: 3.10 or higher
- **MCP library**: `pip install mcp`
- **Bud**: Configured and processed with `bud configure && bud process`

---

## Usage Notes

### Read-Only Operations
The MCP server is strictly read-only. It never modifies Bud's data or index files. All tools are marked with `readOnlyHint: true`.

### Log Management
- Logs are stored separately from Bud's data in `~/mcp_logs/`
- Log files are never automatically deleted
- Users must manage log cleanup themselves
- A typical log file might grow to several MB over time

### Session Behavior
- Each MCP connection gets a unique session ID
- Session IDs persist across CLI restarts
- History is only available within the same session
- Creating a new session (e.g., with `--session-id`) creates a fresh history

### Performance Notes
- The index is lazy-loaded on first access
- Session lifespan keeps the index warm
- First query after cold start may be slower
- Subsequent queries in the same session are faster

---

## Quick Start for Claude

When Claude connects to the MCP server, use this pattern:

```javascript
// Initial session check
const context = await mcp.callTool("bud_context", { history_size: 10 });

if (context.is_new_session) {
  // New conversation - start fresh
  console.log("Starting new MCP session:", context.session_id);
} else {
  // Resuming - check history for recent calls
  console.log("Resuming session:", context.session_id);
  console.log("Recent calls:", context.history.length);
}

// Before making a tool call, check history for duplicates
const shouldCall = !context.history.some(
  h => h.tool_name === "bud_recall" &&
       JSON.stringify(h.parameters) === JSON.stringify({query, k})
);

if (shouldCall) {
  const result = await mcp.callTool("bud_recall", { query, k });
  // Process result...
}
```

---

*Last updated: 2026-03-22*
