# SHL Assessment Recommender

The SHL catalog contains 400+ Individual Test Solutions spanning 10 assessment types. Hiring managers rarely know the right vocabulary to navigate keyword search — they know the role, not the product name. The recommender bridges this by turning a natural-language conversation into a grounded, catalog-validated shortlist.

The system operates as a stateless FastAPI service. Every POST /chat call carries the full conversation history; no server-side session state is stored. This matches the spec and makes the service trivially scalable.

API Link: https://shl-recommender-qttx.onrender.com/docs


---

## Architecture

```
POST /chat
    │
    ▼
Input validation + guard layer
    │
    ▼
Constraint extraction
    │
    ▼
FAISS semantic retrieval + keyword filtering
    │
    ▼
Context-aware LLM recommendation generation (Groq)
    │
    ▼
Recommendation validation against catalog
    │
    ▼
Structured ChatResponse


## Quick Start

### 1. Prerequisites

```bash
python 3.11+
pip install -r requirements.txt
```



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
| LLM (main) | `Groq llama-3.1-8b-instant` | Best reasoning, 1K output |
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
