"""
FastAPI Backend for Ayurvedic Agentic RAG (Production Grade)
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, AsyncGenerator
import uuid
from datetime import datetime
import torch
import os
import json
import asyncio
from dotenv import load_dotenv

from agentic_rag_core import RobustAyurvedicRAG

# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------

load_dotenv()

LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
print("🧠 LangSmith enabled:", LANGSMITH_ENABLED)

print("LangSmith Project:", os.getenv("LANGCHAIN_PROJECT"))
print("LangSmith Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))


# -----------------------------------------------------------------------------
# LOAD CONFIG
# -----------------------------------------------------------------------------

CONFIG_PATH = os.getenv("RAG_CONFIG_PATH", "rag_config.json")

if not os.path.exists(CONFIG_PATH):
    raise RuntimeError("rag_config.json not found")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

PINECONE_API_KEY = config["pinecone_api_key"]
PINECONE_INDEX = config["pinecone_index"]

MODEL_NAME = (
    config.get("production_model")
    if torch.cuda.is_available()
    else config.get("local_model", "google/flan-t5-small")
)

DEVICE = "GPU" if torch.cuda.is_available() else "CPU"

print("✅ Config Loaded")
print(f"   Model: {MODEL_NAME}")
print(f"   Index: {PINECONE_INDEX}")
print(f"   Device: {DEVICE}")

# -----------------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------------

app = FastAPI(
    title="🌿 Ayurvedic Agentic RAG API",
    description="""
Production-grade Agentic RAG system for Ayurveda.

Features:
- Conversational Memory
- Agentic Retrieval & Reranking
- Streaming (SSE)
- LangSmith Tracing
- Confidence Scoring
""",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------

rag: Optional[RobustAyurvedicRAG] = None
sessions: Dict[str, Dict] = {}

# -----------------------------------------------------------------------------
# STARTUP / SHUTDOWN
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Load model ONCE at startup
    """
    global rag

    print("\n" + "=" * 70)
    print("🚀 INITIALIZING AYURVEDIC AGENTIC RAG")
    print("=" * 70)

    rag = RobustAyurvedicRAG(
        pinecone_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX,
        model_name=MODEL_NAME,
    )

    print("✅ RAG system loaded")
    print("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 API shutting down")

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

def get_rag() -> RobustAyurvedicRAG:
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    return rag

# -----------------------------------------------------------------------------
# SCHEMAS
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., example="What are the symptoms of pitta imbalance?")
    session_id: Optional[str] = Field(
        None, description="Client session ID (UUID)"
    )
    use_memory: bool = Field(
        True, description="Enable conversational memory"
    )


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    sources: List[Dict]
    confidence: float
    reasoning: List[str]
    intent: str
    entities: List[str]
    response_time_seconds: float


# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "Ayurvedic Agentic RAG",
        "status": "online",
        "model": MODEL_NAME,
        "device": DEVICE,
        "langsmith": LANGSMITH_ENABLED,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": rag is not None,
        "active_sessions": len(sessions),
        "model": MODEL_NAME,
        "index": PINECONE_INDEX,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag: RobustAyurvedicRAG = Depends(get_rag),
):
    """
    Non-streaming chat endpoint
    """

    session_id = request.session_id or str(uuid.uuid4())

    sessions.setdefault(
        session_id,
        {
            "created_at": datetime.now().isoformat(),
            "message_count": 0,
        },
    )

    try:
        result = await asyncio.to_thread(
            rag.chat,
            request.query,
            session_id=session_id,
            use_memory=request.use_memory,
            verbose=False,
        )

        sessions[session_id]["message_count"] += 1
        sessions[session_id]["last_active"] = datetime.now().isoformat()

        return ChatResponse(
            session_id=session_id,
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            intent=result["intent"],
            entities=result["entities"],
            response_time_seconds=result["response_time_seconds"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    rag: RobustAyurvedicRAG = Depends(get_rag),
):
    """
    Streaming SSE endpoint (token-by-token)
    """

    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            for token in rag.generate_stream(prompt=request.query):
                yield f"data: {token}\n\n"
                await asyncio.sleep(0)  # allow event loop

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        rag.reset_conversation(session_id)
        del sessions[session_id]
        return {"status": "deleted"}

    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/stats")
async def stats():
    return {
        "total_sessions": len(sessions),
        "total_messages": sum(s["message_count"] for s in sessions.values()),
    }

# -----------------------------------------------------------------------------
# LOCAL RUN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("\n🌐 Starting FastAPI server...")
    print("📘 Docs → http://localhost:8000/docs")
    print("=" * 70)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
