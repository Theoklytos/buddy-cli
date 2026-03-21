# Bud RAG - Architecture & Pipeline Diagram

## High-Level System Overview

```
                          +---------------------+
                          |    bud configure     |
                          |  (~/.config/bud/     |
                          |    config.yaml)      |
                          +----------+----------+
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
     +--------+--------+   +--------+--------+   +---------+---------+
     |   LLM Provider  |   | Embed Provider  |   |   Pipeline Cfg    |
     |                  |   |                 |   |                   |
     | - ollama         |   | - ollama        |   | - chunk_min_tok   |
     | - openai (grok)  |   | - openai        |   | - chunk_max_tok   |
     | - anthropic      |   |                 |   | - schema_threshold|
     +---------+--------+   +--------+--------+   +---------+---------+
               |                      |                      |
               +----------+-----------+----------+-----------+
                          |                      |
                          v                      v
                    +-----+------+        +------+------+
                    | LLMClient  |        |EmbedClient  |
                    | .complete()|        | .embed()    |
                    +-----+------+        +------+------+
                          |                      |
         +----------------+----------------------+----------------+
         |                           |                            |
         v                           v                            v
+--------+---------+    +------------+-----------+    +-----------+----------+
|  bud discover    |    |     bud process        |    |    bud query         |
|  (optional)      |    |     (main pipeline)    |    |    (retrieval)       |
+--------+---------+    +------------+-----------+    +-----------+----------+
         |                           |                            |
         v                           v                            v
   See DISCOVER              See PROCESS                   See QUERY
   flow below                flow below                    flow below
```

---

## Main Pipeline: `bud process`

```
 RAW CONVERSATION JSON FILES  (data_dir/*.json)
 ================================================
 [{"chat_messages": [{"sender":"human","content":[...]}, ...]}]

         |
         | parse_all()
         v
 +-------+------------------------------------------+
 |  STAGE 1: PARSE  (bud.stages.parse)              |
 |                                                   |
 |  For each .json file:                             |
 |    - Extract chat_messages array                  |
 |    - Handle content blocks:                       |
 |      [text] [thinking] [tool_use]                 |
 |    - Load memory context (memories_*.json)        |
 |    - Build structured conversations               |
 |                                                   |
 |  Output: parsed/conversations_N.jsonl             |
 +-------+------------------------------------------+
         |
         |  Each conversation = {id, turns[], source_file, ...}
         v
 +-------+------------------------------------------+
 |  STAGE 2: CHUNK  (bud.stages.chunk)              |
 |                                                   |
 |  For each conversation:                           |
 |                                                   |
 |  +-----------+     +-------------+                |
 |  | Prompt    | --> | System      |                |
 |  | Preset    |     | Prompt      |                |
 |  | (md file) |     | + Schema    |                |
 |  +-----------+     | + Discovery |                |
 |                    | (optional)  |                |
 |                    +------+------+                |
 |                           |                       |
 |                    +------v------+                |
 |                    |  LLMClient  |                |
 |                    |  .complete()|                |
 |                    +------+------+                |
 |                           |                       |
 |                    JSON response:                 |
 |                    - chunks[]: turn ranges + tags  |
 |                    - schema_proposals[]            |
 |                                                   |
 |  Tags per chunk:                                  |
 |    geometry:  linear|recursive|spiral|...         |
 |    coherence: tight|loose|fragmented|...          |
 |    texture:   dense|sparse|lyrical|...            |
 |    terrain:   conceptual|emotional|...            |
 |    motifs:    [identity, threshold, ...]           |
 |    type:      exchange|monologue|breakthrough|... |
 |                                                   |
 |  Validation: min_tok <= tokens <= max_tok         |
 |  Fallback:   alternating turn pairs on failure    |
 +-------+------------------------------------------+
         |
         |  chunks[] = [{chunk_id, text, tags, ...}]
         v
 +-------+------------------------------------------+
 |  STAGE 3: EMBED  (bud.stages.embed)              |
 |                                                   |
 |  For each chunk:                                  |
 |    text --> EmbedClient.embed(text[:max_chars])   |
 |         --> [float, float, ..., float]  (vector)  |
 |                                                   |
 |    +--------+     +-----------+                   |
 |    | vector | --> | FAISS     |                   |
 |    | + meta |     | IndexFlat |                   |
 |    +--------+     | IP        |                   |
 |                   +-----------+                   |
 |                                                   |
 |  L2 normalize vectors before indexing             |
 |  Failed chunks --> embed_queue.jsonl (retry)      |
 +-------+------------------------------------------+
         |
         v
 +-------+------------------------------------------+
 |  STAGE 4: SCHEMA EVOLUTION                        |
 |  (bud.lib.schema_manager)                         |
 |                                                   |
 |  schema_proposals from chunking:                  |
 |    {"dimension":"terrain","value":"mythological"}  |
 |                                                   |
 |  propose_candidate() --> track count + examples   |
 |  apply_promotions():                              |
 |    if count >= threshold (default 5):             |
 |      promote to schema.dimensions                 |
 |      increment schema.version                     |
 |      log evolution event                          |
 +-------+------------------------------------------+
         |
         v
 +-------+------------------------------------------+
 |  OUTPUT FILES                                     |
 |                                                   |
 |  output_dir/                                      |
 |    parsed/                                        |
 |      conversations_0.jsonl                        |
 |      conversations_1.jsonl                        |
 |    index/                                         |
 |      chunks.faiss          <-- FAISS binary       |
 |      chunks_metadata.jsonl <-- one dict per line  |
 |    schema.json             <-- evolved schema     |
 |    progress.json           <-- resume tracking    |
 |    embed_queue.jsonl       <-- failed embeds      |
 +--------------------------------------------------+
```

---

## Discovery Flow: `bud discover`

```
 parsed/conversations_N.jsonl
         |
         |  3 sampling modes:
         |
    +----+--------+----------+
    |              |          |
    v              v          v
 RANDOM       BLEND      PROGRESSIVE
 whole-conv   cross-      cursor-based
 sampling     boundary    exhaustive
 (n=6)        slices      (1 slice/file)
              (n=6,w=8)
    |              |          |
    +----+---------+----------+
         |
         v
 +-------+------------------------------------------+
 |  DISCOVERY LOOP (max_iterations=10)               |
 |                                                   |
 |  Each iteration:                                  |
 |                                                   |
 |    samples + current concept_map                  |
 |         |                                         |
 |         v                                         |
 |    +----------+                                   |
 |    | LLM      |  DISCOVERY_SYSTEM_PROMPT:         |
 |    | .complete|  "Analyze structural patterns..." |
 |    +----+-----+                                   |
 |         |                                         |
 |         v                                         |
 |    JSON response:                                 |
 |      boundary_signals:  where chunks end          |
 |      coherence_anchors: what holds chunks         |
 |      chunk_archetypes:  recurring types           |
 |      anti_patterns:     what NOT to do            |
 |      stability_score:   convergence metric        |
 |                                                   |
 |    Merge via apply_update()                       |
 |    Smooth: EMA alpha=0.3                          |
 |                                                   |
 |    if stability >= threshold (0.75): STOP         |
 |                                                   |
 +-------+------------------------------------------+
         |
         v
    discovery_map.json
    (injected into chunking prompts via --with-discovery)
```

---

## Query Flow: `bud query`

```
 User: "How does authentication work?"
         |
         v
 +-------+------------------------------------------+
 |  1. EMBED QUERY                                   |
 |     EmbedClient.embed("How does auth work?")      |
 |     --> query_vector [float x dim]                 |
 +-------+------------------------------------------+
         |
         v
 +-------+------------------------------------------+
 |  2. SEARCH INDEX                                  |
 |     VectorStore.search(query_vector, k=5)         |
 |                                                   |
 |     FAISS IndexFlatIP (cosine similarity)         |
 |                                                   |
 |     Returns top-k chunks:                         |
 |       [{text, tags, score, source_file, ...}]     |
 +-------+------------------------------------------+
         |
         v
 +-------+------------------------------------------+
 |  3. GENERATE ANSWER                               |
 |     Build prompt:                                 |
 |       system: "Answer using retrieved context"    |
 |       user:   context_chunks + original_query     |
 |                                                   |
 |     LLMClient.complete(system, user)              |
 |     --> Generated answer grounded in chunks       |
 +-------+------------------------------------------+
         |
         v
 +-------+------------------------------------------+
 |  4. DISPLAY                                       |
 |     Rank | Score | Source        | Preview         |
 |     -----+-------+--------------+---------        |
 |       1  | 0.92  | conv_3.jsonl | "Auth uses..."  |
 |       2  | 0.87  | conv_7.jsonl | "The token..."  |
 |       ...                                         |
 |                                                   |
 |     Answer: "Authentication works by..."          |
 +--------------------------------------------------+
```

---

## Module Dependency Map

```
                         bud.cli
                        /   |   \
                       /    |    \
                      v     v     v
           bud.config   bud.lib   bud.stages
                        /  |  \      /  |  \  \  \
                       v   v   v    v   v   v  v  v
               embeddings llm store parse chunk embed discover blend
                   |       |    |
                   v       v    v
              [Ollama] [Ollama] [FAISS]
              [OpenAI] [OpenAI]
                       [Claude]

  Shared across stages:
    - model_registry   --> embeddings, cli
    - schema_manager   --> chunk, cli
    - progress         --> cli (resume)
    - prompt_loader    --> cli, chunk
    - errors           --> all modules
    - index            --> cli (path management)
```

---

## Schema Dimension Space

```
  +------------------+---------------------------------------------+
  | Dimension        | Values                                      |
  +------------------+---------------------------------------------+
  | geometry         | linear, recursive, spiral, bifurcating,     |
  |                  | convergent                                  |
  +------------------+---------------------------------------------+
  | coherence        | tight, loose, fragmented, emergent,         |
  |                  | contradictory                               |
  +------------------+---------------------------------------------+
  | texture          | dense, sparse, lyrical, technical, mythic,  |
  |                  | raw                                         |
  +------------------+---------------------------------------------+
  | terrain          | conceptual, emotional, procedural,          |
  |                  | speculative, relational                     |
  +------------------+---------------------------------------------+
  | motifs (multi)   | identity, threshold, resonance,             |
  |                  | system-design, becoming                     |
  +------------------+---------------------------------------------+
  | chunk_type       | exchange, monologue, breakthrough,          |
  |                  | definition, artifact                        |
  +------------------+---------------------------------------------+

  Schema evolves at runtime:
    LLM proposes new values --> candidates{} --> promoted at threshold
```

---

## Resume & Fault Tolerance

```
  progress.json        embed_queue.jsonl       blend_cursor.json
  +---------------+    +-----------------+     +----------------+
  | file: batch:  |    | chunk_id: ...   |     | file: offset   |
  |   done/failed |    | text: ...       |     | file: offset   |
  +---------------+    | tags: ...       |     +----------------+
                       +-----------------+
       |                      |                       |
       v                      v                       v
  Skip completed       Retry failed            Resume progressive
  batches on           embeddings on           discovery from
  --resume             --resume                last position
```

---

## CLI Commands Summary

```
  bud
   |
   +-- configure    Interactive setup wizard
   +-- models       List supported embedding models
   +-- parse        Parse raw conversation JSON into JSONL
   +-- discover     Pattern discovery (optional pre-step)
   |    +-- --blend          Cross-boundary sampling
   |    +-- --progressive    Exhaustive cursor-based
   |    +-- --resume         Continue from last run
   |    +-- --stability 0.75 Early-stop threshold
   +-- chunk [N]    Standalone iterative chunking (N passes)
   |    +-- --concurrency N  Parallel LLM requests
   |    +-- --stability 0.75 Early-stop threshold
   |    +-- --prompt NAME    conversational|factual|mythic|synthesis
   +-- process      Full pipeline (parse -> chunk -> embed)
   |    +-- --resume         Skip completed batches
   |    +-- --batch-size 50  Conversations per batch
   |    +-- --prompt NAME    conversational|factual|mythic|synthesis
   |    +-- --with-discovery Inject discovery map
   +-- query TEXT   Semantic search + LLM answer
   |    +-- --k 5           Top-k results
   +-- status       Show pipeline & index status
   +-- update       Pull latest code, sync dependencies
        +-- --dev            Include dev dependencies
```
