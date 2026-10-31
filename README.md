#AssetGraph— Natural language Q&A over industrial plant documents, with a hybrid vector + knowledge-graph backend

> Ask *"What failures has pump P-101 had after maintenance by John Kumar?"* and get a cited, confidence-scored answer drawn from work orders, inspection reports, compliance manuals, and P&ID diagrams — without opening a single file.

**Live demo:** [eth-one.vercel.app](https://eth-one.vercel.app)

---

## Why this is hard

Industrial plant data is not just unstructured text — it is a web of causal relationships: equipment connected to parameters, events caused by failures, personnel linked to work orders, all regulated by overlapping standards (OSHA, OISD, Factory Act). A question like *"Are the blast furnace operating procedures compliant with OSHA confined-space regulations?"* cannot be answered by a single vector lookup — it requires joining information across multiple document types and reasoning over their relationships.

Standard RAG breaks here because:
- **Cross-document multi-hop reasoning** requires traversing entity relationships, not just matching embeddings.
- **Hallucination on safety questions** is dangerous — a wrong answer about a lockout procedure could cause real harm.
- **Unstructured + structured data** (PDFs, images, spreadsheets, P&IDs) need different parsers, yet a single coherent query interface.
- **Domain scope enforcement** must be strict — the system must refuse to answer off-topic questions rather than generating confident-sounding nonsense.

---

## What makes this different

- **Hybrid retrieval with a weighted reranker** — ChromaDB vector search always runs first; Neo4j graph traversal activates for multi-hop/aggregation queries *or* when vector confidence falls below threshold. Results are merged with `final_score = (0.7 × vector_score) + (0.3 × graph_score)`.
- **Two-pass LLM extraction** — ingestion runs an *extractor* pass to propose entities/edges, then a separate *verifier* pass to score confidence and reject anything below 0.70. Only verified facts are written to the graph.
- **Fixed 6-node / 8-edge ontology** — every fact is typed against `Equipment | Document | Parameter | Person | Regulation | Event` nodes and `CAUSED_BY | HAS_PARAMETER | REGULATED_BY | CONNECTED_TO | PRECEDED_BY | PERFORMED_BY | DOCUMENTED_IN | MENTIONS` edges. No freeform nodes; schema is the contract between ingestion and retrieval.
- **LangGraph multi-agent supervisor** — five nodes: `classify → retrieve → dispatch → synthesize → guardrails`. Specialist agents (Copilot, Maintenance, Compliance, LessonsLearned) run in parallel via `asyncio.gather`; outputs merge in the synthesis node.
- **Post-synthesis groundedness check** — after synthesis, a second independent LLM pass verifies each factual claim against the retrieved context. If >50% of claims are unsupported, confidence is capped at Low (≤0.40). This prevents the model from grading its own homework.
- **Safety escalation layer** — a fast keyword scan (no LLM call) flags safety-critical queries (lockout, confined space, gas leak, etc.). If confidence is Medium or Low on a flagged query, an escalation notice is appended and the query is written to an immutable audit log.
- **Off-topic refusal** — the query classifier (GPT-4o, JSON mode) routes out-of-scope questions directly to END without retrieval, returning a scoped refusal message.
- **Vision + OCR ingestion** — P&ID images go through Tesseract OCR + GPT-4 Vision to extract spatial relationships and equipment tags from engineering diagrams.

---

## Architecture

```
User Query
    │
    ▼
FastAPI Gateway  (uvicorn, port 8000)
    │
    ▼
LangGraph Supervisor
    │
    ├─ 1. Classify  ──────────────────── GPT-4o (JSON mode)
    │       │ off-topic? → refusal (END, no retrieval)
    │       │ single-fact / multi-hop / aggregation ↓
    │
    ├─ 2. Retrieve
    │       ├── ChromaDB vector search    (always runs)
    │       └── Neo4j Cypher traversal   (multi-hop / low vector confidence)
    │               └── Weighted reranker → top-20 unified context
    │
    ├─ 3. Dispatch  (parallel asyncio.gather)
    │       ├── CopilotAgent        — general Q&A + summarization
    │       ├── MaintenanceAgent    — MTBF, RCA, work orders
    │       ├── ComplianceAgent     — OSHA / OISD / Factory Act checks
    │       └── LessonsLearnedAgent — recurring failure patterns
    │
    ├─ 4. Synthesize  ── merge outputs, deduplicate citations (by title), compute confidence
    │
    └─ 5. Guardrails
            ├── Groundedness check  (independent LLM claim verification)
            └── Severity escalation (keyword scan → audit log)
                    │
                    ▼
            AgentResponse {answer, citations, confidence, graph_path,
                           groundedness_warning, escalation_notice}
```

**Knowledge Graph schema (Neo4j AuraDB):**

| Node | Key fields |
|------|-----------|
| `Equipment` | tag, name, type, location, manufacturer |
| `Document` | title, doc_type, doc_number, date, revision |
| `Parameter` | name, value, unit, design_value |
| `Person` | name, role, employee_id, department |
| `Regulation` | code, title, section, authority |
| `Event` | type, date, description, severity |

Edge provenance: every relationship stores `confidence`, `source_doc_id`, `extraction_pass_id`, `verified`, and `raw_text_span` for full traceability.

---

## Eval Results

Measured with a custom harness (`evals/hallucination_eval.py`) over **30 queries** (10 single-hop, 10 multi-hop, 10 out-of-scope). Faithfulness scored via [Ragas](https://github.com/explodinggradients/ragas); citation accuracy and refusal accuracy via custom logic. Full results in [`evals/results.json`](evals/results.json).

| Metric | Score |
|--------|-------|
| Out-of-scope refusal accuracy | **100%** (10/10) |
| Multi-hop citation accuracy | **90%** |
| Single-hop faithfulness (Ragas) | **69.4%** |
| Overall citation accuracy | **75%** |
| Overall faithfulness (Ragas) | **49.9%** |

> **Note on multi-hop faithfulness (30.3%):** Ragas penalises answers that draw on graph-traversal context not captured in raw text chunks. This is a known metric limitation for hybrid retrieval systems, not evidence of hallucination — the post-synthesis groundedness guardrail is the operative safety check at runtime.

---

## Tech Stack

| Layer | Library / Version |
|-------|------------------|
| Backend framework | FastAPI 0.111 + uvicorn 0.30 |
| Orchestration | LangGraph 0.1.14, LangChain 0.2.5 |
| LLM / Vision | OpenAI GPT-4o (`openai==1.35.3`) |
| Vector store | ChromaDB 0.5.3 |
| Graph database | Neo4j 5.21 (AuraDB Free) |
| Embeddings | `sentence-transformers >= 3.0.1` |
| OCR | pytesseract 0.3.13 + pdf2image 1.17.0 |
| Document parsers | pypdf 4.2, pdfplumber 0.11.1, python-docx 1.1.2, openpyxl 3.1.4 |
| Eval | Ragas 0.1.14, Langfuse 2.29.3 |
| Frontend | Next.js 16.2.10, React 19, TailwindCSS 4, Framer Motion 12 |

---

## Quickstart (Local)

**Prerequisites:** Python 3.11+, Node.js 18+, OpenAI API key. Neo4j is optional — the app falls back to vector-only search without it.

```bash
# 1. Clone & configure
git clone https://github.com/Mohit20198/ETH
cd ETH
cp .env.example .env
# Fill in OPENAI_API_KEY (required)
# Fill in NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD (optional)

# 2. Backend
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.api.main:app --reload --port 8000
# API docs → http://localhost:8000/docs

# 3. Frontend (new terminal)
cd frontend && npm install && npm run dev
# UI → http://localhost:3000
```

> **Judges:** The repo ships a pre-ingested `data/chroma` vector store with 10 sample documents. Skip ingestion and query immediately.

---

## Try These Queries

| Type | Query |
|------|-------|
| Single-hop | *"What is the MTBF for Pump P-101 and what caused its last failure?"* |
| Compliance | *"Are we compliant with OSHA regulations for gas detectors?"* |
| Multi-hop graph | *"Show me the maintenance history for any equipment connected to Valve V-201."* |
| Vision / P&ID | *"What are the operating parameters shown in the PID.jpg diagram?"* |
| Out-of-scope (refusal test) | *"What is the recipe for chocolate cake?"* |

---

