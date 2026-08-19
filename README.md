# Agentic RAG Assistant

An adaptive, multimodal Agentic RAG system: PDF ingestion (text + image
captioning) → chunking → Qdrant vector storage → a LangGraph agent that
retrieves, grades document relevance, falls back to web search, generates
grounded answers with citations, checks for hallucination, checks answer
usefulness, and self-corrects with bounded retries.

## Architecture

```
backend/
  agent/
    graph/        LangGraph state, nodes, and graph assembly
    chains/        Graders (relevance, hallucination, usefulness) + generation
    ingestion/     PDF extraction, image captioning, chunking
    retrievers/     Qdrant vector store, retriever, web search tool
    embeddings/     Embedding model factory
    llm_provider.py LLM client factory
  api/            FastAPI app (/chat, /ingest, /health, /sources)
  models/         Pydantic schemas
  config/         Typed settings (pydantic-settings)
  scripts/        Manual smoke-test scripts
frontend/         Next.js chat UI
```

## Local Setup

**Backend:**
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r backend/requirements.txt
copy .env.example .env       # then fill in your real keys
uvicorn backend.api.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`.

## Environment Variables

See `.env.example`. Required: `GEMINI_API_KEY`. Optional: `TAVILY_API_KEY`
(web search fallback).

## Deployment

**Backend → HuggingFace Spaces (Docker SDK):**
1. Create a new Space, SDK = Docker.
2. Push this repo (the `Dockerfile` at the root builds the backend).
3. In Space Settings → Repository secrets, add `GEMINI_API_KEY` and
   `TAVILY_API_KEY`.
4. Note: local Qdrant storage is ephemeral on free-tier Spaces (resets on
   restart). For persistence, switch `QDRANT_MODE=cloud` and point to a
   Qdrant Cloud cluster.

**Frontend → Vercel:**
1. Import the `frontend/` folder as a new Vercel project.
2. Set env var `NEXT_PUBLIC_API_URL` to your deployed backend's URL
   (e.g. `https://your-space.hf.space`).
3. Deploy.

## Tech Stack

FastAPI, LangGraph, LangChain, Qdrant, Next.js, Google Gemini (via
OpenAI-compatible proxy), PyMuPDF.
