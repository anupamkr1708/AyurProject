"""
🌿 Ayurvedic AI Assistant – Production Streamlit Frontend
Compatible with Agentic RAG + FastAPI backend
Run: streamlit run app.py
"""

import streamlit as st
import requests
import uuid
import time
from typing import Dict, List

# =============================================================================
# CONFIG
# =============================================================================

API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 90  # seconds

st.set_page_config(
    page_title="🌿 Ayurvedic AI Assistant",
    page_icon="🌿",
    layout="wide",
)

# =============================================================================
# STYLES
# =============================================================================

st.markdown(
    """
<style>
.big-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #1B5E20;
    text-align: center;
    padding: 1.8rem;
    background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
    border-radius: 18px;
    margin-bottom: 2rem;
}

.user-msg {
    background: #E3F2FD;
    padding: 1.2rem;
    border-radius: 14px;
    margin: 1rem 0;
    border-left: 6px solid #2196F3;
}

.assistant-msg {
    background: #F1F8E9;
    padding: 1.2rem;
    border-radius: 14px;
    margin: 1rem 0;
    border-left: 6px solid #4CAF50;
}

.conf-high { color: #2E7D32; font-weight: bold; }
.conf-medium { color: #EF6C00; font-weight: bold; }
.conf-low { color: #C62828; font-weight: bold; }

.source-box {
    background: #FAFAFA;
    padding: 0.8rem;
    border-radius: 10px;
    border-left: 4px solid #9CCC65;
    margin-bottom: 0.6rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# SESSION STATE
# =============================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict] = []

if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = False

# =============================================================================
# BACKEND HELPERS
# =============================================================================


def check_backend():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return True, r.json()
    except Exception:
        return False, None


def send_chat(query: str):
    payload = {
        "query": query,
        "session_id": st.session_state.session_id,
        "use_memory": True,
    }

    r = requests.post(
        f"{API_URL}/chat",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if r.status_code != 200:
        raise RuntimeError(r.text)

    return r.json()


def stream_chat(query: str):
    payload = {
        "query": query,
        "session_id": st.session_state.session_id,
        "use_memory": True,
    }

    with requests.post(
        f"{API_URL}/chat/stream",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                token = decoded.replace("data: ", "")
                if token == "[DONE]":
                    break
                yield token


def confidence_badge(conf: float):
    if conf >= 0.7:
        return "conf-high", "🟢 High"
    elif conf >= 0.5:
        return "conf-medium", "🟡 Medium"
    return "conf-low", "🔴 Low"


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🌿 Ayurvedic AI")

    healthy, health = check_backend()

    if healthy:
        st.success("Backend Online")
        st.caption(f"Model: {health.get('model')}")
        st.caption(f"Sessions: {health.get('active_sessions')}")
    else:
        st.error("Backend Offline")
        st.code("cd backend && uvicorn main:app --reload")

    st.markdown("---")

    st.checkbox(
        "⚡ Streaming Response",
        value=st.session_state.use_streaming,
        key="use_streaming",
    )

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages.clear()
        st.rerun()

    st.markdown("---")

    st.markdown("### 💡 Example Questions")
    for q in [
        "What is pitta dosha?",
        "Treatment for anxiety in Ayurveda",
        "Diet for pitta imbalance",
        "Daily dinacharya routine",
    ]:
        if st.button(q, use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

# =============================================================================
# MAIN UI
# =============================================================================

st.markdown(
    '<div class="big-title">🌿 Ayurvedic AI Assistant</div>', unsafe_allow_html=True
)

st.caption("Agentic RAG • Classical Texts • Confidence Scoring • Conversational Memory")

# =============================================================================
# CHAT HISTORY
# =============================================================================

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-msg'><b>You</b><br>{msg['content']}</div>",
            unsafe_allow_html=True,
        )
    else:
        cls, label = confidence_badge(msg["confidence"])
        st.markdown(
            f"""
            <div class='assistant-msg'>
            <b>Assistant</b> — <span class='{cls}'>{label} ({msg['confidence']:.1%})</span>
            <br><br>{msg['content']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class='source-box'>
                        <b>{s['source']}</b> (Page {s['page']})<br>
                        Score: {s['score']:.2f}<br>
                        <small>{s['text_preview']}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# =============================================================================
# INPUT
# =============================================================================

st.markdown("---")

user_query = st.text_area(
    "Ask your question",
    height=90,
    value=st.session_state.pop("pending_query", ""),
)

send = st.button("🚀 Ask", disabled=not healthy)

if send and user_query.strip():

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("🧠 Thinking..."):
        try:
            if st.session_state.use_streaming:
                placeholder = st.empty()
                full_answer = ""

                for token in stream_chat(user_query):
                    full_answer += token
                    placeholder.markdown(full_answer)

                result = {
                    "answer": full_answer,
                    "confidence": 0.5,
                    "sources": [],
                }
            else:
                result = send_chat(user_query)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "confidence": result.get("confidence", 0.5),
                    "sources": result.get("sources", []),
                }
            )

        except Exception as e:
            st.error(str(e))

    st.rerun()

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("Ayurvedic AI Assistant • Agentic RAG • For educational purposes only")
