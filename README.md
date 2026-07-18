
# 📖 Gospel RAG — LDS Doctrinal Search & Study Assistant

A Retrieval-Augmented Generation system for exploring LDS doctrine with confidence. Gospel RAG combines deep-scraped data from the **Bible Dictionary** and **Topical Guide** with hybrid semantic + keyword search, so every answer is grounded in and cited to real source material — no hallucinated doctrine, ever.

Ask a question in plain language ("What does the Topical Guide say about faith?") and get an answer synthesized by an LLM, backed by the actual passages it drew from.

---

##  Highlights

- **Grounded answers** — every response is generated from retrieved source chunks and cited back to the original entry, not the model's own memory.
- **Hybrid retrieval** — combines FAISS semantic vector search with BM25 keyword search for both conceptual and exact-term matching.
- **Fast by design** — vector and keyword indexes are loaded into memory once at startup, not re-read from disk on every query.
- **Clean chat UI** — a Next.js + Tailwind interface with client-side chat history, no full-page reloads.

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Frontend | [Next.js](https://nextjs.org/) (TypeScript / React) + [Tailwind CSS](https://tailwindcss.com/) |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) (Python) |
| Orchestration | [LangChain](https://www.langchain.com/) |
| Semantic Search | [FAISS](https://github.com/facebookresearch/faiss) |
| Keyword Search | BM25 |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | OpenAI `gpt-4o-mini` |
| Python tooling | [`uv`](https://github.com/astral-sh/uv) + `pyproject.toml` |

---

##  Architecture

The system is built as a four-stage pipeline, from raw scraped text to an in-browser conversation:

```
 1. Scraping                2. Chunking & Vectorizing        3. API                       4. UI
 ┌──────────────────┐      ┌─────────────────────────┐      ┌─────────────────────┐      ┌──────────────────────┐
 │ Bible Dictionary  │      │ Split scraped text into  │      │ FastAPI serves a     │      │ Next.js chat interface│
 │ Topical Guide     │─────▶│ chunks, embed with       │─────▶│ /query endpoint.     │─────▶│ (React state, no full │
 │ Topics & Questions│      │ text-embedding-3-small,  │      │ FAISS + BM25 indexes │      │ page reloads) renders │
 │ (raw HTML/text →  │      │ build FAISS + BM25       │      │ are loaded into      │      │ streamed, cited       │
 │  data/raw/)        │      │ indexes (data/index/)    │      │ memory at startup    │      │ answers.              │
 └──────────────────┘      └─────────────────────────┘      │ via FastAPI lifespan │      └──────────────────────┘
                                                              │ and never re-read     │
                                                              │ from disk per query.  │
                                                              └─────────────────────┘
```

1. **Scraping** — Source content (Bible Dictionary, Topical Guide, and Topics & Questions) is scraped and stored as raw text/HTML under `data/raw/`.
2. **Chunking & Vectorizing** — Raw documents are split into semantically coherent chunks, embedded with OpenAI's `text-embedding-3-small`, and written to on-disk FAISS and BM25 indexes under `data/index/`.
3. **API** — A FastAPI backend loads both indexes into memory once at startup (via an async lifespan handler), then serves retrieval + generation over a `/query` REST endpoint. LangChain orchestrates the hybrid retrieval and the `gpt-4o-mini` generation step, always grounding responses in retrieved, cited source chunks.
4. **UI** — A Next.js/Tailwind frontend provides a chat interface, managing conversation history entirely in client-side React state and calling the FastAPI backend for each query.

---

## Local Setup

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (Python package/dependency manager)
- [Node.js](https://nodejs.org/) + `npm`
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone the repository

```bash
git clone <https://github.com/SegunOladeinde/Gospel-topics-RAG.git>

cd lds-rag-system
```

### 2. Configure environment variables

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. Set up the backend (Python / FastAPI)

This project uses `uv` exclusively for Python dependency management — do not use `pip` or `requirements.txt`.

```bash
# Install dependencies from pyproject.toml
uv sync

# Run the scraper to populate data/raw/ (if not already populated)
uv run src/fetch_data.py

# Start the FastAPI server
uv run uvicorn src.backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 4. Set up the frontend (Next.js)

The frontend lives entirely in `src/frontend/` and uses `npm`.

```bash
cd src/frontend
npm install
npm run dev
```

The app will be available at `http://localhost:3000`.

> The FastAPI backend is configured with CORS to accept requests from `http://localhost:3000`, so both servers can run side by side during local development.

---

## 📁 Project Structure

```
lds-rag-system/
├── data/
│   ├── raw/            # Scraped source data (Bible Dictionary, Topical Guide, Topics & Questions)
│   └── index/           # Persisted FAISS + BM25 indexes
├── src/
│   ├── fetch_data.py    # Scraping entry point
│   ├── backend/         # FastAPI application, retrieval + generation logic
│   └── frontend/        # Next.js chat application
├── tests/                # Test suite
├── pyproject.toml        # Python dependencies (managed via uv)
└── .env                  # OpenAI API key (not committed)
```

---

## A Note on Doctrinal Accuracy

This is an independent study tool and is not affiliated with or endorsed by The Church of Jesus Christ of Latter-day Saints. All responses are grounded in the scraped source datasets and include citations back to the original material — the system is designed to surface and summarize existing reference content, not to generate new doctrinal claims.
