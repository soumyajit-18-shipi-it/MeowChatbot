# ============================================================
#  MEOWBOT AI X
#  Advanced RAG + NLP + Analytics Chatbot
#  Stack:
#  Streamlit · Groq/OpenAI · ChromaDB · SentenceTransformers
#  PyMuPDF · Transformers · Plotly · Scikit-learn
# ============================================================

import os
import fitz
import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import nltk

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from collections import Counter
from analytics_enhanced import (
    extract_advanced_keywords,
    analyze_reading_complexity,
    extract_entities,
    analyze_emotional_tone,
    detect_document_tone,
    create_semantic_map,
    generate_ai_insights,
    format_insights_text,
    plot_advanced_keywords,
    plot_semantic_map,
    plot_sentiment_gauge,
    plot_document_complexity
)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ============================================================
# ENV
# ============================================================

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MeowBot",
    page_icon="🐱",
    layout="wide",
)

# ============================================================
# CSS
# ============================================================

GLOBAL_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;700&display=swap');

:root {
    --bg: #0e0b1a;
    --panel: #16122a;
    --accent: #c084fc;
    --pink: #f472b6;
    --text: #e9d5ff;
    --muted: #a78bfa;
    --border: rgba(192,132,252,0.2);
    --green: #4ade80;
}

html, body, [data-testid="stApp"] {
    background: var(--bg);
    color: var(--text);
    font-family: 'Nunito', sans-serif;
}

[data-testid="stSidebar"] {
    background: #0a0815;
    border-right: 1px solid var(--border);
}

/* ── SIDEBAR LABELS ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--muted) !important;
    font-family: 'Nunito', sans-serif;
}

/* ── SLIDERS ── */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: var(--accent) !important;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: rgba(22,18,42,0.6);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 12px 16px;
    backdrop-filter: blur(6px);
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid var(--border) !important;
    border-radius: 24px !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(192,132,252,0.15) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    transition: transform 0.15s, opacity 0.15s;
}

.stButton > button:hover {
    opacity: 0.88 !important;
    transform: scale(1.02);
}

/* ── DIVIDER ── */
hr {
    border-color: var(--border) !important;
}

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 1.8rem 1rem 1rem;
}

.hero h1 {
    font-family: 'Fredoka One', cursive;
    font-size: 2.8rem;
    background: linear-gradient(90deg, #c084fc, #f472b6, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
    margin: 0;
}

@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.hero p {
    color: var(--muted);
    margin: 0.4rem 0 0 0;
    font-size: 1rem;
}

.hero .paws {
    font-size: 1.4rem;
    letter-spacing: 6px;
    margin-top: 4px;
    opacity: 0.5;
}

/* ── CARD ── */
.card {
    background: rgba(22,18,42,0.7);
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 18px;
}

/* ── STATUS BADGE ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: var(--green);
    font-weight: 700;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.75); }
}

/* ── CAT BG ── */
.chat-bg-wrapper {
    position: relative;
}

.cat-watermark {
    position: fixed;
    bottom: 90px;
    right: 30px;
    opacity: 0.055;
    pointer-events: none;
    z-index: 0;
    font-size: 260px;
    line-height: 1;
    filter: grayscale(1);
    user-select: none;
}

/* ── MEOW REPLY ── */
.meow-reply {
    font-family: 'Fredoka One', cursive;
    font-size: 1.6rem;
    background: linear-gradient(90deg, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── TIP BOX ── */
.tip-box {
    background: rgba(192,132,252,0.08);
    border: 1px solid rgba(192,132,252,0.25);
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 8px;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    background: rgba(22,18,42,0.5) !important;
}

/* ── SELECTBOX / INPUTS ── */
[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.06) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
}

/* ── PLOTLY CHARTS ── */
.js-plotly-plot .plotly .bg {
    fill: transparent !important;
}

</style>

<!-- Fixed cat watermark behind chat -->
<div class="cat-watermark">🐱</div>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ============================================================
# SESSION
# ============================================================

defaults = {
    "messages": [],
    "pdf_text": "",
    "chunks": [],
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 1.0,
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# API KEY
# ============================================================

def get_api_key():
    return (
        os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )

# ============================================================
# CLIENT
# ============================================================

def build_client(api_key):
    if api_key.startswith("gsk_"):
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return OpenAI(api_key=api_key)

# ============================================================
# MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def load_sentiment_model():
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

embedding_model = None
sentiment_pipeline = None

def ensure_models_loaded():
    global embedding_model, sentiment_pipeline
    if embedding_model is None:
        embedding_model = load_embedding_model()
    if sentiment_pipeline is None:
        sentiment_pipeline = load_sentiment_model()

# ============================================================
# VECTOR DB
# ============================================================

@st.cache_resource
def get_chroma_collection():
    client = chromadb.Client()
    return client.get_or_create_collection(name="meowbot_pdf")

collection = get_chroma_collection()

# ============================================================
# PDF
# ============================================================

def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)

# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(chunks):
    ensure_models_loaded()
    if embedding_model is None:
        st.error("Embedding model not loaded")
        return np.zeros((len(chunks), 384))
    try:
        return embedding_model.encode(chunks)
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return np.zeros((len(chunks), 384))

# ============================================================
# VECTOR STORE
# ============================================================

def store_chunks(chunks):
    try:
        # Get all existing IDs and delete them
        existing = collection.get()
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception as e:
        st.warning(f"Could not clear old chunks: {e}")
    
    embeddings = create_embeddings(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(query, n_results=4):
    ensure_models_loaded()
    if embedding_model is None:
        return ["Embedding model not available. Please refresh the page."]
    try:
        query_embedding = embedding_model.encode([query])
        results = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
        return results["documents"][0]
    except Exception as e:
        st.error(f"Error retrieving chunks: {e}")
        return ["Error retrieving relevant information."]

# ============================================================
# NLP
# ============================================================

def tokenize_text(text):
    try:
        return word_tokenize(text)
    except:
        return text.split()

def get_word_frequencies(tokens):
    return Counter(tokens)

def bag_of_words(text):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform([text])
    return vectorizer, X

# ============================================================
# ANALYTICS DASHBOARD - ENHANCED
# ============================================================

def render_analytics_dashboard(text, embedding_model, sentiment_pipeline):
    """Render comprehensive AI analytics dashboard"""
    
    st.markdown("---")
    st.markdown("### 🤖 AI Document Intelligence Dashboard")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Content Analysis", "💭 Sentiment & Tone", "🗺️ Semantic Map", "✨ AI Insights"])
    
    with tab1:
        st.subheader("Content Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Advanced Keywords
            try:
                with st.spinner("Extracting keywords..."):
                    keywords = extract_advanced_keywords(text, embedding_model, top_n=12)
                    keyword_freq = [text.lower().count(kw.lower()) for kw in keywords]
                    
                    fig = plot_advanced_keywords(keywords, keyword_freq)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Keyword extraction failed: {e}")
        
        with col2:
            # Document Complexity
            try:
                complexity = analyze_reading_complexity(text)
                st.metric("📖 Readability Score", f"{complexity['flesch_score']}", complexity['complexity'])
                st.metric("⏱️ Reading Time", complexity['reading_time'])
                st.metric("📏 Avg Sentence Length", f"{complexity['avg_sentence_length']} words")
            except Exception as e:
                st.warning(f"Complexity analysis failed: {e}")
        
        # Named Entity Recognition
        try:
            st.subheader("Named Entities Detected")
            entities = extract_entities(text)
            
            if entities and "error" not in entities:
                cols = st.columns(min(5, len([e for e in entities.values() if e])))
                entity_types = {"ORG": "🏢", "PERSON": "👤", "GPE": "📍", "PRODUCT": "📦", "EVENT": "📅"}
                
                col_idx = 0
                for entity_type, entity_list in entities.items():
                    if entity_list and col_idx < len(cols):
                        with cols[col_idx]:
                            emoji = entity_types.get(entity_type, "📌")
                            st.write(f"**{emoji} {entity_type}**")
                            for entity in entity_list:
                                st.caption(entity)
                        col_idx += 1
        except Exception as e:
            st.warning(f"Entity recognition unavailable: {e}")
    
    with tab2:
        st.subheader("Sentiment & Emotional Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment Analysis
            try:
                sentiment = analyze_emotional_tone(text, sentiment_pipeline)
                if sentiment:
                    fig = plot_sentiment_gauge(sentiment)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**Confidence Score:** {sentiment['confidence']}%")
            except Exception as e:
                st.warning(f"Sentiment analysis failed: {e}")
        
        with col2:
            # Writing Tone
            try:
                tone = detect_document_tone(text)
                st.metric("📝 Writing Tone", tone.get("primary_tone", "Unknown"))
                st.metric("🎓 Formality Score", f"{tone.get('formality_score', 0)}/100")
                
                st.markdown("**Tone Indicators:**")
                for tone_type, count in tone.get("indicators", {}).items():
                    st.write(f"- {tone_type.title()}: {count} occurrences")
            except Exception as e:
                st.warning(f"Tone analysis failed: {e}")
    
    with tab3:
        st.subheader("Semantic Document Clustering")
        
        try:
            # Create semantic embeddings
            embedding_model.encode(text[:100])  # Test embedding
            
            # For proper clustering, we'd use chunks - this is simplified
            st.info("🗺️ Semantic map requires PDF chunks for visualization. Upload a PDF to see the interactive semantic map.")
            
        except Exception as e:
            st.warning(f"Semantic mapping unavailable: {e}")
    
    with tab4:
        st.subheader("✨ AI-Generated Document Insights")
        
        try:
            with st.spinner("Generating AI insights..."):
                insights = generate_ai_insights(text, embedding_model, sentiment_pipeline)
                insights_text = format_insights_text(insights)
                st.markdown(insights_text)
                
                # Suggested questions
                st.markdown("**💡 Suggested Questions to Ask:**")
                themes = insights.get("main_themes", [])
                if themes:
                    for i, theme in enumerate(themes[:3], 1):
                        st.write(f"{i}. What are the key points about {theme}?")
        except Exception as e:
            st.warning(f"Insight generation failed: {e}")

# ============================================================
# SEMANTIC CLUSTERING - ENHANCED
# ============================================================

def render_semantic_clustering(chunks, embeddings):
    """Render advanced semantic clustering visualization"""
    try:
        if len(chunks) < 3:
            st.info("Not enough chunks for clustering (minimum 3 required)")
            return
        
        # Create semantic map
        cluster_df = create_semantic_map(embeddings, chunks, n_clusters=min(5, len(chunks)))
        
        if cluster_df is not None:
            # Interactive semantic map
            fig = plot_semantic_map(cluster_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster summary
            st.markdown("**Cluster Summary:**")
            for cluster_name in cluster_df['cluster_name'].unique():
                cluster_chunks_list = cluster_df[cluster_df['cluster_name'] == cluster_name]['text_preview'].tolist()
                with st.expander(f"📌 {cluster_name} ({len(cluster_chunks_list)} chunks)"):
                    for i, chunk_text in enumerate(cluster_chunks_list, 1):
                        st.caption(f"{i}. {chunk_text}")
        else:
            st.warning("Could not create semantic clusters")
    except Exception as e:
        st.error(f"Clustering error: {e}")

# ============================================================
# PROMPT
# ============================================================

def build_prompt(user_input):
    relevant_chunks = retrieve_relevant_chunks(user_input)
    context = "\n\n".join(relevant_chunks)
    return f"""You are MeowBot — a friendly AI research assistant.

Use the retrieved context below to answer accurately.

CONTEXT:
{context}

QUESTION:
{user_input}
"""

# ============================================================
# LLM RESPONSE
# ============================================================

def generate_response(client, model, messages):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
        top_p=st.session_state.top_p,
    )
    return response.choices[0].message.content

# ============================================================
# MEOW EASTER EGG
# ============================================================

def is_meow(text: str) -> bool:
    """Return True if the message is just variations of 'meow'."""
    cleaned = text.strip().lower().rstrip("!")
    return cleaned in {"meow", "meeow", "meooow", "meoow", "meeeow", "mrow", "mrrow"}

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🐱 MeowBot</h1>
    <p>Advanced Retrieval-Augmented Neural Intelligence System</p>
    <div class="paws">🐾 🐾 🐾</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

api_key = get_api_key()
model = "llama-3.3-70b-versatile"

with st.sidebar:

    st.markdown("""
<div style="font-family:'Fredoka One',cursive;font-size:1.4rem;
background:linear-gradient(90deg,#c084fc,#f472b6);-webkit-background-clip:text;
-webkit-text-fill-color:transparent;padding-bottom:4px;">
🐱 MeowBot
</div>
<div class="status-badge">
<div class="status-dot"></div> Online
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("⚙️ LLM Parameters")

    st.session_state.temperature = st.slider("🌡️ Temperature", 0.0, 2.0, 0.7, step=0.05)
    st.session_state.max_tokens  = st.slider("📏 Max Tokens",  256, 4096, 2048, step=128)
    st.session_state.top_p       = st.slider("🎯 Top P",       0.1, 1.0, 1.0, step=0.05)

    st.divider()
    st.subheader("📄 Upload PDF")

    uploaded_pdf = st.file_uploader("Drop a PDF to enable RAG", type=["pdf"])

    if uploaded_pdf:
        with st.spinner("🐱 Processing PDF..."):
            text   = extract_pdf_text(uploaded_pdf)
            st.session_state.pdf_text = text
            chunks = chunk_text(text)
            st.session_state.chunks   = chunks
            store_chunks(chunks)

        st.success("✅ PDF processed!")
        st.markdown(f"""
<div class="card">
<strong>📊 PDF Stats</strong><br>
🧩 Chunks: <code>{len(chunks)}</code><br>
🔤 Characters: <code>{len(text):,}</code><br>
📝 Words: <code>{len(text.split()):,}</code>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
<div class="tip-box">
💡 <strong>Easter Egg:</strong> Send <code>meow</code> to the bot — it speaks cat! 🐾
</div>
""", unsafe_allow_html=True)

    st.write("")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:
    avatar = "🐱" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input("Ask MeowBot... or just say meow 🐾")

if prompt:

    if not api_key:
        st.error("🔑 Please set GROQ_API_KEY or OPENAI_API_KEY in your .env file")
        st.stop()

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # ── MEOW EASTER EGG ──────────────────────────────────────
    if is_meow(prompt):
        meow_response = '<span class="meow-reply">meow</span> 🐾'
        st.session_state.messages.append({"role": "assistant", "content": meow_response})
        with st.chat_message("assistant", avatar="🐱"):
            st.markdown(meow_response, unsafe_allow_html=True)

    # ── NORMAL RAG RESPONSE ───────────────────────────────────
    else:
        final_prompt = build_prompt(prompt)
        api_messages = [
            {"role": "system", "content": "You are MeowBot — a friendly and helpful AI research assistant. Be concise and accurate."},
            {"role": "user",   "content": final_prompt},
        ]

        with st.spinner("🐱 MeowBot is thinking..."):
            client   = build_client(api_key)
            response = generate_response(client, model, api_messages)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar="🐱"):
            st.markdown(response)

# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

if st.session_state.pdf_text:
    st.divider()
    
    ensure_models_loaded()
    render_analytics_dashboard(
        st.session_state.pdf_text[:5000],
        embedding_model,
        sentiment_pipeline
    )
    
    if len(st.session_state.chunks) >= 3:
        st.markdown("---")
        st.markdown("### 🗺️ Semantic Clustering Analysis")
        embeddings = create_embeddings(st.session_state.chunks[:30])
        render_semantic_clustering(st.session_state.chunks[:30], embeddings)