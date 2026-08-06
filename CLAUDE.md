
# Project: LDS Doctrinal RAG System

## Project Architecture
* **Backend:** FastAPI (Python) serving a REST API.
* **Frontend:** Next.js (TypeScript/React/Tailwind) running independently.
* **AI & Search:** LangChain, FAISS (semantic vector search), BM25 (keyword search).
* **LLMs:** OpenAI `gpt-4o-mini` and `text-embedding-3-small`.

## Tooling & Execution Rules
1. **Python Management:** Use `uv` and `pyproject.toml` exclusively. NEVER use `pip` or `requirements.txt`. Add packages via `uv add <package>`. Run scripts via `uv run <script>`.
2. **Next.js Management:** Use `npm`. The frontend lives entirely in `src/frontend/`.
3. **Backend Performance:** The FAISS and BM25 indexes MUST be loaded into memory globally upon startup using FastAPI's `@asynccontextmanager` lifespan. Never load indexes from disk during a `/query` request.
4. **CORS:** FastAPI must use `CORSMiddleware` to allow requests from `http://localhost:3000`.
5. **Frontend State:** Next.js must manage chat history strictly using React `useState` and avoid full page reloads.
6. **Data Grounding:** All AI responses must cite the scraped LDS datasets. Do not allow hallucinated doctrines.