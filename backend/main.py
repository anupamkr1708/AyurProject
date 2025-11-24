"""
FastAPI Backend for Ayurvedic RAG
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import uuid
from datetime import datetime

try:
    from agentic_rag_core import RobustAyurvedicRAG
    USING_MODULE = True
except ImportError:
    USING_MODULE = False
    print("  Import failed - will initialize inline")

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

print(" Loading configuration...")
with open('rag_config.json', 'r') as f:
    config = json.load(f)

PINECONE_API_KEY = config['pinecone_api_key']
PINECONE_INDEX = config['pinecone_index']
MODEL_NAME = config['model_name']

print(f" Config loaded")
print(f"   Model: {MODEL_NAME}")
print(f"   Index: {PINECONE_INDEX}")

# ============================================================================
# INITIALIZE RAG SYSTEM
# ============================================================================

print("\n Initializing RAG system...")
print("   (This will download the model - takes 2-3 min first time)")

if USING_MODULE:
    rag = RobustAyurvedicRAG(
        pinecone_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX,
        model_name=MODEL_NAME
    )
else:
    # If module import failed, copy the full class here
    # OR: Create a separate agentic_rag_core.py file
    print("  Please create agentic_rag_core.py with RobustAyurvedicRAG class")
    exit(1)

print(" RAG system ready!\n")

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="🌿 Ayurvedic RAG API",
    description="Robust Agentic RAG for Ayurvedic Knowledge",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage
sessions = {}


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    use_memory: bool = True

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


@app.get("/")
def root():
    return {
        "service": "Ayurvedic Agentic RAG",
        "version": "3.0.0",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "active_sessions": len(sessions),
        "model": MODEL_NAME,
        "index": PINECONE_INDEX
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main chat endpoint"""
    
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "message_count": 0
        }
    
    try:
        result = rag.chat(
            request.query,
            use_memory=request.use_memory,
            verbose=False
        )
        
        sessions[session_id]["message_count"] += 1
        sessions[session_id]["last_active"] = datetime.now().isoformat()
        
        return ChatResponse(
            session_id=session_id,
            query=result['query'],
            answer=result['answer'],
            sources=result['sources'],
            confidence=result['confidence'],
            reasoning=result['reasoning'],
            intent=result['intent'],
            entities=result['entities'],
            response_time_seconds=result['response_time_seconds']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        rag.reset_conversation()
        del sessions[session_id]
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/stats")
def get_stats():
    return {
        "total_sessions": len(sessions),
        "total_messages": sum(s.get("message_count", 0) for s in sessions.values())
    }

if __name__ == "__main__":
    import uvicorn
    print("\n Starting FastAPI server...")
    print(" Docs: http://localhost:8000/docs")
    print("="*70)
    uvicorn.run(app, host="0.0.0.0", port=8000)