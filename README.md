# SHL Assessment Recommender

Conversational agent that takes a hiring manager from a vague intent to a grounded
shortlist of SHL Individual Test Solution assessments through multi-turn dialogue.

Built for the SHL Labs AI Intern take-home assignment.

---

## Architecture

```
POST /chat (stateless history)
        │
        ▼
   Guard layer ──► blocked? → short-circuit reply
        │
        ▼
Constraint extraction (claude-haiku: fast, cheap)
        │
        ▼
FAISS semantic search + keyword filter → catalog context
        │
        ▼
Main LLM reasoning (claude-sonnet: full context + catalog)
        │
        ▼
Parse <RECOMMENDATIONS> block → validate URLs against catalog
        │
        ▼
ChatResponse { reply, recommendations, end_of_conversation }
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Stateless API | Full history per request; no server session; trivially scalable |
| Two-stage retrieval | FAISS for meaning + keyword for exact matches; better than either alone |
| URL integrity guarantee | All URLs validated against Set[str] loaded from catalog at startup; hallucination structurally impossible |
| Pre-LLM guard layer | Injection/off-topic caught before spending tokens |
| Separate constraint extraction | Cheap haiku call extracts structured filters; sonnet focuses on reasoning |
| Turn budget enforcement | Agent warned at turn 6; hard cut at 8 |

---

## Quick Start

### 1. Prerequisites

```bash
python 3.11+
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env

```

### 3. Place catalog

Put `catalog.json` in `data/`. The file should be the SHL product catalog JSON
(array of assessment objects with `entity_id`, `name`, `link`, `keys`, etc.).

### 4. Build FAISS index

```bash
python -m scripts.build_index
```

This creates `data/index.faiss` and `data/index_ids.npy`. Run once; re-run
if catalog changes.

### 5. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker:

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 --env-file .env shl-recommender
```

---

## API Reference

### `GET /health`

```json
{"status": "ok"}
```

Returns `200` when ready. First call allows up to 2 minutes for cold start.

### `POST /chat`

**Request:**

```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**

```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**Schema rules (non-negotiable):**
- `recommendations` is `[]` when clarifying or refusing
- `recommendations` has 1–10 items when committed to a shortlist
- `end_of_conversation` is `true` only when task is complete
- Every `url` comes from the catalog (validated server-side)

**Constraints:**
- Max 8 turns per conversation (400 error if exceeded)
- 30-second timeout per call
- `role` must be `"user"` or `"assistant"`; last message must be `"user"`

---

## Test Types

| Code | Category |
|---|---|
| A | Ability & Aptitude |
| B | Biodata & Situational Judgment |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |
| K | Knowledge & Skills |
| P | Personality & Behavior |
| S | Simulations |

---

---

## Evaluation

The agent is graded on:

1. **Hard evals (must pass)**
   - Schema compliance on every response
   - All recommendation URLs from catalog
   - 8-turn cap honoured

2. **Recall@10** — fraction of relevant assessments appearing in top-10 recommendations

3. **Behavior probes** (binary pass/fail)
   - Refuses off-topic requests
   - Does not recommend on vague turn 1
   - Honours edits/refinements
   - No hallucinated URLs
   - Handles comparison queries with catalog data only

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| LLM (main) | `claude-sonnet-4-20250514` | Best reasoning, 1K output |
| LLM (extraction) | `claude-haiku-4-5-20251001` | Fast + cheap constraint parsing |
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, fast, good general retrieval |
| Vector store | FAISS (IndexFlatIP) | CPU-friendly, exact cosine, no infra |
| API framework | FastAPI | Async, auto-schema, production-ready |
| Deployment | Render / Docker | Free tier, cold start < 2min |

---

## File Structure

```
shl-recommender/
├── app/
│   ├── main.py          # FastAPI endpoints
│   ├── models.py        # Pydantic schemas
│   ├── agent.py         # Conversation orchestration
│   ├── retrieval.py     # FAISS + keyword search
│   ├── catalog.py       # Catalog loader + URL validation
│   ├── prompts.py       # All prompt templates
│   └── guards.py        # Pre-LLM guard layer
├── data/
│   ├── catalog.json     # SHL catalog (you provide)
│   ├── index.faiss      # Built by build_index.py
│   └── index_ids.npy    # Built by build_index.py
├── scripts/
│   └── build_index.py   # One-time index builder
├── tests/
│   ├── test_api.py
│   ├── test_agent.py
│   ├── test_retrieval.py
│   ├── test_guards.py
│   ├── test_catalog.py
│   └── traces/
│       └── sample_trace.json
├── Dockerfile
├── requirements.txt
└── .env.example
```
