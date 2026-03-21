<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/FAISS-vector_search-FF6F00?style=for-the-badge" alt="FAISS"/>
  <img src="https://img.shields.io/badge/Ollama-LLM-000000?style=for-the-badge" alt="Ollama"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

<h1 align="center">bud</h1>

<p align="center">
  <strong>A personal conversation RAG pipeline that turns your AI conversation history into searchable structural memory.</strong>
</p>

<p align="center">
  <em>Parse. Discover. Chunk. Embed. Query.</em>
</p>

---

## What is Bud?

Bud ingests exported AI conversation data, discovers structural patterns in how your conversations unfold, and builds a FAISS vector index you can query with natural language. Unlike generic RAG systems, Bud understands the **geometry** of conversation — spirals, breakthroughs, recursive loops — and tags every chunk with rich structural metadata.

```
conversations_*.json  ──►  parse  ──►  discover  ──►  chunk  ──►  embed  ──►  query
                            │            │              │           │           │
                         extract      find patterns   LLM splits   vectors   semantic
                         turns        in structure    with tags    in FAISS   search
```

## Features

- **Structural tagging** — Every chunk is tagged across five dimensions: geometry, coherence, texture, terrain, and motifs
- **Pattern discovery** — An iterative LLM loop samples your archive and builds a concept map of structural patterns before chunking begins
- **Schema evolution** — The tagging schema grows at runtime as the LLM discovers new dimension values
- **Three sampling modes** — Whole-conversation, random cross-boundary blend, or progressive cursor-based exhaustive coverage
- **Iterative refinement** — Multi-pass chunking with self-review: the LLM critiques its own output and improves on the next pass
- **Concurrent chunking** — `-c 6` fires 6 LLM requests in parallel for cloud providers; defaults to sequential for local
- **Sentence-aware truncation** — Never cuts text mid-word; truncates at sentence or word boundaries
- **Resume & fault tolerance** — Checkpointed progress, failed embedding queue with retry, blend cursor persistence
- **Provider-agnostic** — Ollama, OpenAI-compatible (Grok), or Anthropic Claude for LLM; Ollama or OpenAI for embeddings
- **Model-aware limits** — Chunk sizes and embedding windows auto-configure from a model registry

## Quickstart

```bash
# Clone and install
git clone https://github.com/Theoklytos/buddy-cli.git
cd buddy-cli
make dev            # creates .venv, installs with dev deps

# Configure (sets data dir, output dir, LLM/embedding models)
bud configure

# Run the pipeline
bud parse                    # parse raw conversations → JSONL
bud discover 10              # 10 iterations of pattern discovery
bud chunk 5 -c 4             # 5 refinement passes, 4 concurrent
bud process -c 6             # or run everything end-to-end

# Query your memory
bud query "How does the auth system work?"
```

## Installation

**Requirements:** Python 3.10+, a running [Ollama](https://ollama.com) instance (or cloud LLM API).

```bash
# Option A: Make (recommended)
make dev

# Option B: Manual
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Option C: Script
bash install.sh --dev
```

## Commands

| Command | Purpose |
|---------|---------|
| `bud configure` | Interactive setup wizard |
| `bud parse` | Parse raw conversation JSON into JSONL |
| `bud discover [N]` | Run N iterations of pattern discovery |
| `bud chunk [N] [-c C]` | Iterative chunking with N passes, C concurrent |
| `bud process [-c C]` | Full pipeline: parse + chunk + embed |
| `bud query "text"` | Semantic search with LLM-generated answer |
| `bud status` | Show pipeline and index status |
| `bud models` | List supported embedding models |
| `bud update` | Pull latest code and sync dependencies |

### Key Flags

```
-c, --concurrency N    Concurrent LLM requests (default: 1 = sequential)
-p, --prompt PRESET    conversational | factual | mythic | synthesis
-D, --discover N       Inline discovery before chunking (in bud process)
--blend                Cross-boundary sampling mode (in bud discover)
--progressive          Exhaustive cursor-based sampling (in bud discover)
```

## Input Format

Bud expects **Claude conversation exports** — the JSON files you can download from [claude.ai](https://claude.ai) via Settings → Export Data. Each file contains one or more conversations with the structure:

```json
[{
  "chat_messages": [
    {"sender": "human", "content": [{"type": "text", "text": "..."}]},
    {"sender": "assistant", "content": [{"type": "text", "text": "..."}]}
  ]
}]
```

Place your exported `conversations_*.json` files in your configured `data_dir` before running the pipeline.

> Support for other export formats (ChatGPT, etc.) is not currently implemented.

## Architecture

```
bud/
├── cli.py              # Click CLI — all commands
├── config.py           # YAML config (~/.config/bud/config.yaml)
├── lib/
│   ├── llm.py          # LLM client (Ollama, OpenAI-compat, Claude)
│   ├── embeddings.py   # Embedding client (Ollama, OpenAI)
│   ├── store.py        # FAISS IndexFlatIP vector store
│   ├── model_registry.py  # Model → dimension/token limits
│   ├── schema_manager.py  # Tag schema + evolution
│   ├── prompt_loader.py   # Markdown template engine
│   └── errors.py       # Error hierarchy
├── stages/
│   ├── parse.py        # JSON → structured JSONL
│   ├── discover.py     # Iterative pattern discovery
│   ├── chunk.py        # LLM-driven chunking (sequential + batch)
│   ├── chunk_refine.py # Multi-pass refinement with self-review
│   ├── embed.py        # Vectorize chunks → FAISS
│   ├── blend.py        # Cross-boundary sampling + cursor
│   └── index.py        # Output path management
└── prompts/            # Chunking prompt presets (Markdown)
```

### Pipeline Data Flow

```
Raw JSON  ──►  parse_all()  ──►  parsed/*.jsonl
                                      │
                         ┌────────────┤
                         ▼            ▼
                   discover()    chunk_conversation()
                         │       or chunk_conversations_batch()
                         ▼            │
                discovery_map.json    ▼
                    (optional)   embed_chunks()
                         │            │
                         └─────►──────▼
                              FAISS index
                              + metadata.jsonl
```

### Structural Dimensions

Every chunk is tagged across five dimensions:

| Dimension | Example Values |
|-----------|---------------|
| **Geometry** | linear, recursive, spiral, bifurcating, convergent |
| **Coherence** | tight, loose, fragmented, emergent |
| **Texture** | dense, sparse, lyrical, technical, mythic, raw |
| **Terrain** | conceptual, emotional, procedural, speculative |
| **Motifs** | identity, threshold, resonance, system-design, becoming |

Plus a `chunk_type`: exchange, monologue, breakthrough, definition, artifact.

The schema evolves at runtime — when the LLM proposes new values repeatedly, they get promoted into the active schema.

## Configuration

Config lives at `~/.config/bud/config.yaml`:

```yaml
data_dir: /path/to/conversation/exports
output_dir: /path/to/bud/output

llm:
  provider: ollama          # ollama | grok | claude
  base_url: http://localhost:11434
  model: gemma3:latest

embeddings:
  provider: ollama          # ollama | openai
  base_url: http://localhost:11434
  model: nomic-embed-text
```

Environment variables in config values are expanded: `ngrok_token: ${NGROK_TOKEN}`

## Testing

```bash
make test                # full suite (pytest -v)
make test-fast           # quick run (pytest -q)

# Single test
.venv/bin/pytest tests/test_chunk.py::test_function_name -v
```

All tests mock external services. No running LLM or Ollama required.

## Origin

Bud was built on a Samsung Galaxy running [Termux](https://termux.dev), using a Claude Pro account and a mix of free cloud services and local models running on native Android hardware. The author is a carpenter with a deep interest in AI — not a software engineer. The project grew through iterative prompting: a human with a clear vision of what they needed, and a model doing the implementation. It exists because the right tool for the job didn't.

## License

MIT
