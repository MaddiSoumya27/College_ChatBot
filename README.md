# BVRIT Hyderabad RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for BVRIT HYDERABAD College of Engineering for Women, built for the GenAI & Agentic AI Engineering Day 4 Lab.

---

## Project Structure

```
ChatBot_Building/
├── kb/                          # Source markdown files
│   ├── 00_college_overview.md
│   ├── 01_admissions.md
│   ├── 02_departments.md
│   ├── 03_placements.md
│   ├── 04_facilities_research_governance.md
│   └── 05_faculty_directory.md
├── bvrith_chroma_db/            # ChromaDB vector store (auto-created by ingest.py)
├── BVRITH_Knowledge_Base.docx   # Merged 8-section grounding doc (built by build_kb_docx.py)
├── build_kb_docx.py             # Phase 0 — Build the .docx from KB markdown files
├── ingest.py                    # Phase 1 — Chunk, embed, persist to ChromaDB
├── rag_pipeline.py              # Phase 2+3 — Retrieval + Grounded Generation
├── app.py                       # Phase 4 — Streamlit Chat UI
├── generate_test_cases.py       # Phase 5 — Generate 20+ test cases
├── run_eval.py                  # Phase 5 — Run evaluation + RAGAS scoring
├── test_cases.json              # Generated test cases (auto-created)
├── eval_report.json             # Evaluation results (auto-created)
├── eval_report.md               # Human-readable evaluation report (auto-created)
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

---

## Run Order

### Phase 0 — Build Knowledge Base Document

```bash
python build_kb_docx.py
```

Creates `BVRITH_Knowledge_Base.docx` with 8 properly-headed sections.

### Phase 1 — Ingest & Index

```bash
python ingest.py
```

Chunks the docx, embeds with `text-embedding-3-small`, persists to `./bvrith_chroma_db`.

Verify the index:
```bash
python ingest.py --check
```

### Phase 4 — Run the Chatbot

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Phase 5 — Generate Test Cases

```bash
python generate_test_cases.py
```

Creates `test_cases.json` with 25+ cases across 8 evaluation dimensions.

### Phase 5 — Run Evaluation

```bash
python run_eval.py
```

Runs all test cases against the live chatbot, scores with GPT-4o as judge, outputs `eval_report.md` and `eval_report.json`.

---

## Evaluation Dimensions

| # | Dimension | Cases |
|---|---|---|
| 01 | Functional | 3 |
| 02 | Quality | 3 |
| 03 | Safety | 2 |
| 04 | Security | 2 |
| 05 | Robustness | 3 |
| 06 | Performance | 2 |
| 07 | Context / Multi-turn | 4 |
| 08 | RAGAS | 3 |

---

## Architecture

```
User Question
     │
     ▼
Streamlit UI (app.py)
     │
     ▼
run_rag_pipeline() ──► ChromaDB retriever (top-k=5)
     │                        │
     │                        ▼
     │                 Retrieved chunks with
     │                 section/subsection metadata
     │
     ▼
GPT-4o mini (grounded generation)
  - Must cite every fact: [Section, Chunk N]
  - Must refuse if not in context
  - Must stay in scope (BVRIT topics only)
     │
     ▼
Response + Citations + Refused flag
     │
     ▼
Streamlit UI (renders response, shows sources badge)
```
