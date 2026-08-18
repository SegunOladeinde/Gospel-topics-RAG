# LDS Gospel RAG System

## Project Overview
The LDS Gospel RAG (Retrieval-Augmented Generation) System is an intelligent, context-aware conversational AI that answers doctrinal questions by retrieving and synthesizing knowledge directly from approved LDS Gospel source texts (like the standard works, General Conference, and Gospel Topics essays). Designed to strictly ground its responses in primary source material, the system acts as a knowledgeable, faithful study companion that cites its sources and mitigates LLM hallucinations.

## Architecture
The platform is built on a high-performance **dual-stack architecture**:
- **Frontend**: A modern, responsive **Next.js** web application utilizing modular React components and Tailwind CSS. The interface features a premium "Smoked Glass" (glassmorphism) UI aesthetic built over dynamic mesh and radial backgrounds.
- **Backend**: A robust, async-first **Python FastAPI** service. It utilizes LangChain for LLM orchestration, providing a type-safe, enterprise-grade generative engine.

## Key Features
-  **AI Intent Router**: Dynamically classifies incoming user queries (`CHITCHAT`, `RAG_FOLLOWUP`, or `RAG_NEW_TOPIC`) via a 3-way branching classifier. This enables intelligent history-rewriting and avoids unnecessary database retrievals. 
-  **FAISS/BM25 Ensemble Retriever**: Combines semantic meaning (FAISS embeddings) with exact keyword matching (BM25) to provide highly accurate, dual-pronged context retrieval against the vast text corpus.
-  **Dual-Theme Glassmorphism UI**: Seamless Light and Dark mode integration utilizing `next-themes`, ensuring frosted glass interfaces and floating elements remain visually stunning across all device preferences.

## Security Measures
The backend has been deeply refactored and audited to guarantee production-grade security:
- **HMAC API Key Validation**: Uses constant-time string comparisons to prevent timing-oracle side-channel attacks during authentication.
- **Payload Size Limits**: Custom Starlette middleware drops requests exceeding 1MB before parsing, neutralizing RAM-exhaustion (DDoS) vectors.
- **SHA-256 Index Validation**: Verifies the integrity of the binary `.pkl` BM25 index on startup, neutralizing potential Remote Code Execution (RCE) via malicious `pickle` injection.
- **Prompt Injection Guards**: Secures LLM structures via strict `typing.Literal` Pydantic bounds and regex-sanitized path citations.

## Environment Setup
Create a `.env` file in the root of the project to configure the environment. Do not commit this file to version control.

```env
# Required: Your OpenAI API key for embeddings and generation
OPENAI_API_KEY="sk-proj-..."

# Required: Shared secret to authenticate the frontend requests to the backend
APP_API_KEY="your-secure-random-string"

# Optional/Prod: SHA-256 checksum of your data/index/bm25.pkl file
# to guarantee database integrity prior to server boot.
BM25_INDEX_SHA256="your-sha256-hash"

# Optional: CORS allowed origins (defaults to http://localhost:3000)
ALLOWED_ORIGINS="http://localhost:3000,https://app.yourdomain.com"
```

## Local Development

### 1. Start the FastAPI Backend
Ensure your Python environment is set up and locked via `uv`. From the root repository directory, start the server:

```bash
uv run uvicorn src.backend.main:app --reload --port 8000
```
*The endpoint will be available at `http://localhost:8000`. Interactive API documentation can be viewed at `http://localhost:8000/docs`.*

### 2. Start the Next.js Frontend
Navigate into the Next.js application directory and run the Node development server:

```bash
cd src/frontend
npm run dev
```
*The user interface will be available at `http://localhost:3000`.*
