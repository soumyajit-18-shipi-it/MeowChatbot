# ============================================================
#  MEOWBOT AI X
#  Advanced RAG + NLP + Analytics Chatbot
#  Stack:
#  Streamlit · Groq/OpenAI · ChromaDB · SentenceTransformers
#  PyMuPDF · Transformers · Plotly · Scikit-learn
# ============================================================

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import fitz
import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
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

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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

/* ── CHAT INPUT — fixed to viewport bottom ── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 999 !important;
    padding: 0.6rem 1.2rem 0.6rem !important;
    background: linear-gradient(
        180deg,
        rgba(14,11,26,0) 0%,
        rgba(14,11,26,0.96) 18%,
        rgba(14,11,26,1) 100%
    ) !important;
    /* Push it right of the sidebar (Streamlit sidebar default width ≈ 336px) */
    margin-left: 336px !important;
}

[data-testid="stChatInput"] > div {
    margin-bottom: 0 !important;
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
}

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

/* ── Extra bottom padding so last message isn't hidden behind fixed bar ── */
[data-testid="stVerticalBlock"] {
    padding-bottom: 90px !important;
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
    return client.get_or_create_collection(
        name="meowbot_pdf",
        metadata={"hnsw:space": "cosine"},
    )

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
        existing = collection.get()
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception as e:
        st.warning(f"Could not clear old chunks: {e}")
    
    embeddings = create_embeddings(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)


RAG_RELEVANCE_THRESHOLD = 0.45
RAG_MAX_CHUNKS = 4


def build_hybrid_system_prompt():
    return """You are MeowBot, a helpful and conversational AI assistant with hybrid RAG behavior.

Use uploaded PDF context only when it is clearly relevant to the user's question.
If the retrieved PDF context is strong and relevant, weave it naturally into the answer and keep the response grounded in the document.
If the retrieved context is weak, tangential, or unrelated, ignore it and answer using your own general knowledge.

Rules:
- Prefer PDF context for document-specific facts, names, numbers, quotes, and summaries.
- Blend PDF context with general knowledge when both are useful.
- Never invent PDF details that are not supported by the retrieved context.
- Never mention retrieval scores, missing context, or internal retrieval failures.
- Never say the answer is not found in the document unless the user explicitly asks about the document contents.
- Stay natural, concise when appropriate, and conversational.
- Always try to help the user directly.
"""


def format_retrieved_context(chunks, scores):
    lines = []
    for index, chunk in enumerate(chunks, 1):
        score = scores[index - 1] if index - 1 < len(scores) else 0.0
        lines.append(f"[Context {index} | relevance={score:.2f}] {chunk}")
    return "\n\n".join(lines)

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(query, n_results=RAG_MAX_CHUNKS, relevance_threshold=RAG_RELEVANCE_THRESHOLD):
    ensure_models_loaded()
    if embedding_model is None:
        return {
            "chunks": [],
            "scores": [],
            "best_score": 0.0,
            "mean_score": 0.0,
            "use_context": False,
        }

    try:
        query_embedding = embedding_model.encode([query])
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "distances"],
        )

        documents = results.get("documents", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        scored_chunks = []
        for document, distance in zip(documents, distances):
            try:
                similarity = 1.0 - float(distance)
            except (TypeError, ValueError):
                similarity = 0.0
            scored_chunks.append((document, max(0.0, similarity)))

        relevant_chunks = [chunk for chunk, score in scored_chunks if score >= relevance_threshold]
        relevant_scores = [score for chunk, score in scored_chunks if score >= relevance_threshold]
        best_score = max((score for _, score in scored_chunks), default=0.0)
        mean_score = float(np.mean(relevant_scores)) if relevant_scores else 0.0

        return {
            "chunks": relevant_chunks,
            "scores": relevant_scores,
            "best_score": best_score,
            "mean_score": mean_score,
            "use_context": bool(relevant_chunks),
        }
    except Exception as e:
        st.error(f"Error retrieving chunks: {e}")
        return {
            "chunks": [],
            "scores": [],
            "best_score": 0.0,
            "mean_score": 0.0,
            "use_context": False,
        }

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
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Content Analysis", "💭 Sentiment & Tone", "🗺️ Semantic Map", "✨ AI Insights"])
    
    with tab1:
        st.subheader("Content Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                with st.spinner("Extracting keywords..."):
                    keywords = extract_advanced_keywords(text, embedding_model, top_n=12)
                    keyword_freq = [text.lower().count(kw.lower()) for kw in keywords]
                    
                    fig = plot_advanced_keywords(keywords, keyword_freq)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Keyword extraction failed: {e}")
        
        with col2:
            try:
                complexity = analyze_reading_complexity(text)
                st.metric("📖 Readability Score", f"{complexity['flesch_score']}", complexity['complexity'])
                st.metric("⏱️ Reading Time", complexity['reading_time'])
                st.metric("📏 Avg Sentence Length", f"{complexity['avg_sentence_length']} words")
            except Exception as e:
                st.warning(f"Complexity analysis failed: {e}")
        
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
            embedding_model.encode(text[:100])
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
        
        cluster_df = create_semantic_map(embeddings, chunks, n_clusters=min(5, len(chunks)))
        
        if cluster_df is not None:
            fig = plot_semantic_map(cluster_df)
            st.plotly_chart(fig, use_container_width=True)
            
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

def build_prompt(user_input, retrieval_result=None):
    messages = [
        {"role": "system", "content": build_hybrid_system_prompt()},
    ]

    if retrieval_result and retrieval_result.get("use_context"):
        context = format_retrieved_context(
            retrieval_result.get("chunks", []),
            retrieval_result.get("scores", []),
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"User question:\n{user_input}\n\n"
                    f"Relevant PDF excerpts:\n{context}\n\n"
                    "Use the PDF excerpts when they help answer the question. "
                    "If the excerpts do not cover part of the answer, rely on your own general knowledge and keep the response natural."
                ),
            }
        )
    else:
        messages.append({"role": "user", "content": user_input})

    return messages

# ============================================================
# LLM RESPONSE
# ============================================================

def generate_response(client, model, user_input, retrieval_result=None):
    messages = build_prompt(user_input, retrieval_result)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
        top_p=st.session_state.top_p,
    )
    return response.choices[0].message.content


def scroll_chat_to_bottom():
        """Scroll the viewport to the end of the latest rendered chat message."""
        components.html(
                """
                <script>
                    const scrollLatestMessage = () => {
                        const messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
                        const lastMessage = messages.length ? messages[messages.length - 1] : null;

                        if (lastMessage) {
                            lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
                            return;
                        }

                        const fallbackTargets = [
                            window.parent.document.querySelector('section.main'),
                            window.parent.document.querySelector('[data-testid="stAppViewContainer"]'),
                            window.parent.document.scrollingElement,
                            window.parent.document.documentElement,
                        ].filter(Boolean);

                        for (const target of fallbackTargets) {
                            try {
                                target.scrollTop = target.scrollHeight;
                            } catch (error) {}
                        }
                    };

                    requestAnimationFrame(() => setTimeout(scrollLatestMessage, 50));
                </script>
                """,
                height=0,
        )

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

chat_tab, analytics_tab = st.tabs(["💬 Chat", "📊 Document Analysis"])

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

with chat_tab:
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

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # ── MEOW EASTER EGG ──────────────────────────────────────
        if is_meow(prompt):
            meow_response = '<span class="meow-reply">meow</span> 🐾'
            st.session_state.messages.append({"role": "assistant", "content": meow_response})
            with st.chat_message("assistant", avatar="🐱"):
                st.markdown(meow_response, unsafe_allow_html=True)
            scroll_chat_to_bottom()

        # ── NORMAL RAG RESPONSE ───────────────────────────────────
        else:
            retrieval_result = retrieve_relevant_chunks(prompt)

            with st.spinner("🐱 MeowBot is thinking..."):
                client   = build_client(api_key)
                response = generate_response(client, model, prompt, retrieval_result)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant", avatar="🐱"):
                st.markdown(response)
            scroll_chat_to_bottom()

with analytics_tab:
    # ============================================================
    # ANALYTICS DASHBOARD
    # ============================================================

    if st.session_state.pdf_text:
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
    else:
        st.info("Upload a PDF in the sidebar to unlock the document intelligence dashboard.")