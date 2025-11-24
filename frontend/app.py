"""
Streamlit Frontend for Ayurvedic RAG
Run: streamlit run app.py
"""

import streamlit as st
import requests
from datetime import datetime
import uuid

# ============================================================================
# CONFIG
# ============================================================================

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="🌿 Ayurvedic AI Assistant",
    page_icon="🌿",
    layout="wide"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .big-title {
        font-size: 3rem;
        font-weight: 800;
        color: #2E7D32;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    
    .user-msg {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1.5rem;
        border-radius: 15px 15px 5px 15px;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
    }
    
    .assistant-msg {
        background: linear-gradient(135deg, #F1F8E9 0%, #DCEDC8 100%);
        padding: 1.5rem;
        border-radius: 15px 15px 15px 5px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    
    .conf-high { color: #4CAF50; font-weight: bold; }
    .conf-medium { color: #FF9800; font-weight: bold; }
    .conf-low { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api():
    """Check if API is online"""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200, r.json()
    except:
        return False, None

def send_message(query: str):
    """Send message to API"""
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "use_memory": True
            },
            timeout=60
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_confidence_badge(conf: float):
    """Get confidence badge"""
    if conf >= 0.7:
        return "conf-high", "🟢 High", conf
    elif conf >= 0.5:
        return "conf-medium", "🟡 Medium", conf
    else:
        return "conf-low", "🔴 Low", conf

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <img src='https://img.icons8.com/color/96/ayurveda.png' width='100'/>
        <h2 style='color: #2E7D32;'>Ayurvedic AI</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System status
    st.subheader("🔌 Status")
    is_healthy, health = check_api()
    
    if is_healthy:
        st.success("✅ Online")
        if health:
            st.metric("Sessions", health.get('active_sessions', 0))
            st.metric("Vectors", "26,844")
    else:
        st.error("❌ Offline")
        st.info("Start backend:\n```bash\ncd backend\npython main.py\n```")
    
    st.markdown("---")
    
    # Session controls
    st.subheader("🎛️ Controls")
    
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Quick examples
    st.subheader("💡 Examples")
    examples = [
        "What are the three doshas?",
        "Treatment for anxiety",
        "Diet for Pitta imbalance",
        "Daily Ayurvedic routine"
    ]
    
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.example_query = ex
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<div class="big-title">🌿 Ayurvedic AI Assistant 🌿</div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 1rem; color: #666;'>
    <p>Ask questions about Ayurveda and receive expert guidance from classical texts</p>
    <p><em>Powered by Agentic RAG with Multi-Step Reasoning</em></p>
</div>
""", unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f'<div class="user-msg"><strong>You:</strong><br>{msg["content"]}</div>', 
                   unsafe_allow_html=True)
    else:
        cls, label, conf = get_confidence_badge(msg.get('confidence', 0))
        st.markdown(f'''
        <div class="assistant-msg">
            <strong>Assistant:</strong> <span class="{cls}">{label} ({conf:.1%})</span><br><br>
            {msg["content"]}
        </div>
        ''', unsafe_allow_html=True)
        
        # Sources
        if msg.get('sources'):
            with st.expander(f"📚 {len(msg['sources'])} Sources"):
                for i, src in enumerate(msg['sources'], 1):
                    st.markdown(f"**{i}. {src['source']}** (Page {src['page']}) - Score: {src['score']:.2f}")
                    st.caption(src['text_preview'])

# Chat input
st.markdown("---")

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_area(
        "Your Question:",
        height=100,
        placeholder="Ask about doshas, treatments, diet, lifestyle...",
        value=st.session_state.get('example_query', ''),
        key="chat_input"
    )
    
    # Clear example
    if 'example_query' in st.session_state:
        del st.session_state.example_query

with col2:
    st.write("")
    st.write("")
    send_btn = st.button("🚀 Send", type="primary", use_container_width=True, disabled=not is_healthy)

if send_btn and user_input.strip():
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input
    })
    
    # Get response
    with st.spinner("🧠 Thinking..."):
        result = send_message(user_input)
        
        if result:
            st.session_state.messages.append({
                'role': 'assistant',
                'content': result['answer'],
                'confidence': result['confidence'],
                'sources': result['sources']
            })
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🌿 Ayurvedic AI Assistant v3.0</p>
    <p>Agentic RAG • Advanced Reranking • Conversational Memory</p>
    <p><em>For educational purposes only</em></p>
</div>
""", unsafe_allow_html=True)