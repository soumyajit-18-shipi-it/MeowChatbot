# ============================================================
#  SHIPNCHAT
#  Advanced RAG + NLP + Analytics Chatbot
#  Stack:
#  Streamlit · Groq/OpenAI · ChromaDB · SentenceTransformers
#  PyMuPDF · Transformers · Plotly · Scikit-learn
# ============================================================

import os
import time

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
    page_title="ShipNChat",
    page_icon="☕",
    layout="wide",
)

# ============================================================
# CSS
# ============================================================

GLOBAL_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;700&display=swap');

:root {
    --bg: #1f1410;
    --panel: #2b1d16;
    --accent: #c98a5b;
    --cream: #f5e6d3;
    --text: #f3e7d8;
    --muted: #d4a373;
    --border: rgba(201,138,91,0.25);
    --green: #7fb069;
}

html, body, [data-testid="stApp"] {
    background: var(--bg);
    color: var(--text);
    font-family: 'Nunito', sans-serif;
}

[data-testid="stSidebar"] {
    background: #1a110d;
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
    background: rgba(43,29,22,0.7);
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
        rgba(31,20,16,0) 0%,
        rgba(31,20,16,0.96) 18%,
        rgba(31,20,16,1) 100%
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
    background: rgba(245,230,211,0.08) !important;
    border: 1px solid var(--border) !important;
    border-radius: 24px !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(201,138,91,0.18) !important;
}

/* ── Extra bottom padding so last message isn't hidden behind fixed bar ── */
[data-testid="stVerticalBlock"] {
    padding-bottom: 90px !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #8b4513, #d4a373) !important;
    color: #fff8ec !important;
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
    margin: 0;
}

.hero h1 span.title-text {
    background: linear-gradient(90deg, #c98a5b, #f5e6d3, #c98a5b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
    display: inline-block;
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

.hero .beans {
    font-size: 1.4rem;
    letter-spacing: 6px;
    margin-top: 4px;
    opacity: 0.5;
}

/* ── CARD ── */
.card {
    background: rgba(43,29,22,0.75);
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 18px;
}

/* ── STATUS BADGE ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(127,176,105,0.12);
    border: 1px solid rgba(127,176,105,0.35);
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

/* ── COFFEE WATERMARK ── */
.chat-bg-wrapper {
    position: relative;
}

.coffee-watermark {
    position: fixed;
    bottom: 90px;
    right: 30px;
    opacity: 0.06;
    pointer-events: none;
    z-index: 0;
    font-size: 260px;
    line-height: 1;
    filter: grayscale(1);
    user-select: none;
}

/* ── BREW REPLY (signature echo) ── */
.brew-reply {
    font-family: 'Fredoka One', cursive;
    font-size: 1.6rem;
    background: linear-gradient(90deg, #c98a5b, #f5e6d3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── TIP BOX ── */
.tip-box {
    background: rgba(201,138,91,0.10);
    border: 1px solid rgba(201,138,91,0.28);
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
    background: rgba(43,29,22,0.55) !important;
}

/* ── SELECTBOX / INPUTS ── */
[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] input {
    background: rgba(245,230,211,0.08) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
}

/* ── PLOTLY CHARTS ── */
.js-plotly-plot .plotly .bg {
    fill: transparent !important;
}

</style>

<!-- Fixed coffee watermark behind chat -->
<div class="coffee-watermark">☕</div>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ============================================================
# SESSION
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "document_metadata" not in st.session_state:
    st.session_state.document_metadata = {}
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = {}

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
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_embedding_model():
    try:
        return load_model()
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
def get_chroma_client():
    db_path = os.path.join(os.getcwd(), "data", "chroma_db")
    if os.path.exists(db_path) and not os.path.isdir(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)

@st.cache_resource
def get_chroma_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="shipnchat_pdf",
        metadata={"hnsw:space": "cosine"},
    )

collection = get_chroma_collection()

# ============================================================
# PDF
# ============================================================

@st.cache_data
def extract_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)

def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.read()
    return extract_text(pdf_bytes)

# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text, chunk_size=700, chunk_overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = words[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
            
    # Step 8: Reject empty or tiny chunks
    chunks = [c for c in chunks if len(c.strip()) > 50]
    
    # Step 8: Display chunk sizes
    print(f"[DEBUG CHUNKS] Total chunks: {len(chunks)}")
    for idx, c in enumerate(chunks[:5]):
        print(f"[DEBUG CHUNKS] Chunk {idx} length: {len(c)}")
        
    return chunks

# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(chunks):
    ensure_models_loaded()
    if embedding_model is None:
        st.error("Embedding model not loaded")
        return np.zeros((len(chunks), 384))
    try:
        return embedding_model.encode(chunks, batch_size=32, show_progress_bar=False)
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return np.zeros((len(chunks), 384))

# ============================================================
# VECTOR STORE
# ============================================================

def store_chunks(chunks):
    coll = st.session_state.vectorstore or get_chroma_collection()
    try:
        existing = coll.get()
        if existing and existing["ids"]:
            coll.delete(ids=existing["ids"])
    except Exception as e:
        st.warning(f"Could not clear old chunks: {e}")

    embeddings = create_embeddings(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    coll.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)


RAG_RELEVANCE_THRESHOLD = 0.45
RAG_MAX_CHUNKS = 4


def build_hybrid_system_prompt():
    return """You are ShipNChat, a helpful and conversational AI assistant with hybrid RAG behavior.

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
    coll = st.session_state.vectorstore or get_chroma_collection()
    if embedding_model is None or coll is None:
        return {
            "chunks": [],
            "scores": [],
            "best_score": 0.0,
            "mean_score": 0.0,
            "use_context": False,
        }

    try:
        # Step 2: Print model name
        print(f"[DEBUG EMBEDDINGS] Query model: all-MiniLM-L6-v2")
        query_embedding = embedding_model.encode([query])
        results = coll.query(
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

        # Step 4: Remove similarity threshold
        relevant_chunks = [chunk for chunk, score in scored_chunks]
        relevant_scores = [score for chunk, score in scored_chunks]
        best_score = max((score for _, score in scored_chunks), default=0.0)
        mean_score = float(np.mean(relevant_scores)) if relevant_scores else 0.0

        # Step 3: Print retrieval scores
        print("[DEBUG RETRIEVAL] Scored chunks:")
        for idx, (doc, score) in enumerate(scored_chunks):
            print(f"  Chunk {idx} Score: {score:.4f}")
            print(f"  Snippet: {doc[:200]}")

        return {
            "chunks": relevant_chunks,
            "scores": relevant_scores,
            "best_score": best_score,
            "mean_score": mean_score,
            "use_context": bool(relevant_chunks),
            "previews": [c[:300] + "..." for c in relevant_chunks],
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

    # Incorporate previous conversation history to preserve chat memory
    for msg in st.session_state.messages:
        # Exclude the last user message because we will append it formatted below
        if msg["role"] == "user" and msg["content"] == user_input:
            continue
        messages.append({"role": msg["role"], "content": msg["content"]})

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
    elif st.session_state.get("uploaded_filename") is not None:
        # A PDF is loaded, but no relevant chunks were retrieved. Inform the LLM contextually
        messages.append(
            {
                "role": "user",
                "content": (
                    f"User question:\n{user_input}\n\n"
                    "Note: A PDF document is currently loaded, but no relevant context chunks were found for this query. "
                    "If the user is asking about the document, please say: 'PDF loaded, but no relevant context was found.'"
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
# BREW EASTER EGG (coffee-themed signature reply)
# ============================================================

def is_brew(text: str) -> bool:
    """Return True if the message is a coffee-themed signature phrase."""
    cleaned = text.strip().lower().rstrip("!")
    return cleaned in {"brew", "brewing", "espresso", "latte", "coffee", "cappuccino"}


def process_pdf_callback():
    uploaded_file = st.session_state.get("pdf_uploader")
    if uploaded_file is not None:
        start_total = time.time()
        
        # 1. Parsing
        start_parse = time.time()
        pdf_bytes = uploaded_file.read()
        text = extract_text(pdf_bytes)
        parse_time = time.time() - start_parse
        
        # 2. Chunking
        start_chunk = time.time()
        chunks = chunk_text(text, chunk_size=700, chunk_overlap=50)
        chunk_time = time.time() - start_chunk
        
        # Step 1: Verify chunk insertion
        print(f"[DEBUG INDEXING] Number of chunks: {len(chunks)}")
        if chunks:
            print(f"[DEBUG INDEXING] Chunk 0 preview: {chunks[0][:300]}")
            
        # 3. Embedding & Vector Insertion
        start_embed = time.time()
        coll = get_chroma_collection()
        
        # Clear old chunks first, but reuse the collection
        try:
            existing = coll.get()
            if existing and existing["ids"]:
                coll.delete(ids=existing["ids"])
        except Exception as e:
            pass
        
        ensure_models_loaded()
        if embedding_model is not None:
            # Step 2: Verify embeddings model name
            print(f"[DEBUG EMBEDDINGS] Indexing model: all-MiniLM-L6-v2")
            embeddings = embedding_model.encode(chunks, batch_size=32, show_progress_bar=False)
            embed_time = time.time() - start_embed
            
            start_insert = time.time()
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            coll.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)
            insert_time = time.time() - start_insert
            
            # Step 1: Verify chunk count
            print(f"[DEBUG INDEXING] Collection count after insert: {coll.count()}")
            # Step 7: Check collection contents
            try:
                contents = coll.get()
                print("[DEBUG INDEXING] Collection get() documents count:", len(contents.get("documents", [])))
            except Exception as e:
                print("[DEBUG INDEXING] Failed to get contents:", e)
        else:
            embeddings = None
            embed_time = 0.0
            insert_time = 0.0
            
        total_time = time.time() - start_total
        
        st.session_state.pdf_text = text
        st.session_state.chunks = chunks
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.chunk_count = len(chunks)
        st.session_state.vectorstore = coll
        st.session_state.embeddings = embeddings
        st.session_state.document_metadata = {
            "filename": uploaded_file.name,
            "char_count": len(text),
            "word_count": len(text.split()),
        }
        st.session_state.diagnostics = {
            "upload_time": 0.05,
            "parse_time": parse_time,
            "chunk_time": chunk_time,
            "embed_time": embed_time,
            "insert_time": insert_time,
            "total_time": total_time,
        }
    else:
        # Reset PDF state
        st.session_state.pdf_text = ""
        st.session_state.chunks = []
        st.session_state.uploaded_filename = None
        st.session_state.chunk_count = 0
        st.session_state.vectorstore = None
        st.session_state.embeddings = None
        st.session_state.document_metadata = {}
        st.session_state.diagnostics = {}


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <h1>☕ <span class="title-text">ShipNChat</span></h1>
    <p>Advanced Retrieval-Augmented Neural Intelligence System</p>
    <div class="beans">☕ ☕ ☕</div>
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
<div style="font-family:'Fredoka One',cursive;font-size:1.4rem;padding-bottom:4px;display:flex;align-items:center;gap:6px;">
<span>☕</span>
<span style="background:linear-gradient(90deg,#c98a5b,#f5e6d3);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">ShipNChat</span>
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

    uploaded_pdf = st.file_uploader(
        "Drop a PDF to enable RAG",
        type=["pdf"],
        key="pdf_uploader",
        on_change=process_pdf_callback
    )

    # Fallback to process PDF if state is out of sync (e.g. after code reload)
    if uploaded_pdf is not None and st.session_state.get("uploaded_filename") != uploaded_pdf.name:
        process_pdf_callback()
    elif uploaded_pdf is None and st.session_state.get("uploaded_filename") is not None:
        process_pdf_callback()

    if st.session_state.get("uploaded_filename"):
        st.success(
            f"PDF processed successfully. {st.session_state.chunk_count} chunks indexed."
        )
        st.markdown(f"""
<div class="card">
<strong>📊 PDF Stats</strong><br>
🧩 Chunks: <code>{st.session_state.chunk_count}</code><br>
🔤 Characters: <code>{st.session_state.document_metadata.get('char_count', 0):,}</code><br>
📝 Words: <code>{st.session_state.document_metadata.get('word_count', 0):,}</code>
</div>
""", unsafe_allow_html=True)

        # Step 9: Bypass RAG toggle
        bypass_rag = st.checkbox("⚙️ Bypass RAG (Always use first 4 chunks)", value=False, key="bypass_rag")

        if st.session_state.get("diagnostics"):
            with st.expander("🛠️ Diagnostics"):
                diag = st.session_state.diagnostics
                st.write(f"⏱️ **Parse Time**: {diag.get('parse_time', 0.0):.3f}s")
                st.write(f"⏱️ **Chunking Time**: {diag.get('chunk_time', 0.0):.3f}s")
                st.write(f"⏱️ **Embedding Time**: {diag.get('embed_time', 0.0):.3f}s")
                st.write(f"⏱️ **Vector Insert Time**: {diag.get('insert_time', 0.0):.3f}s")
                
                # Step 1 & Step 10: Collection count verification
                coll_count = st.session_state.vectorstore.count() if st.session_state.vectorstore else 0
                st.write(f"🗃️ **Collection Count**: `{coll_count}`")
                st.write(f"🧠 **Embedding Model**: `all-MiniLM-L6-v2`")
                
                if "retrieve_time" in diag:
                    st.write(f"🔍 **Retrieval Time**: {diag['retrieve_time']:.3f}s")
                if "llm_time" in diag:
                    st.write(f"🧠 **LLM Response Time**: {diag['llm_time']:.3f}s")
                st.write(f"⚙️ **Bypass RAG Active**: `{st.session_state.get('bypass_rag', False)}`")
                
                if "retrieved_chunks_count" in diag:
                    st.markdown("---")
                    st.markdown("**🔍 Last Query Retrieval Metrics:**")
                    st.write(f"📄 **Chunks Retrieved**: {diag['retrieved_chunks_count']}")
                    st.write(f"🎯 **Similarity Scores**: {['{:.3f}'.format(s) for s in diag.get('similarity_scores', [])]}")
                    st.write(f"📝 **Context Chars**: {diag.get('context_length', 0)}")
                    
                    # Step 10: Retrieved chunk previews
                    if diag.get("retrieved_previews"):
                        st.markdown("**📄 Retrieved Chunk Previews:**")
                        for idx, preview in enumerate(diag["retrieved_previews"], 1):
                            st.text_area(f"Chunk {idx} (Preview)", preview, height=100, disabled=True, key=f"preview_chunk_{idx}")

    st.divider()

    st.markdown("""
<div class="tip-box">
💡 <strong>Easter Egg:</strong> Send <code>brew</code> to the bot — it serves a fresh reply! ☕
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
        avatar = "☕" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # ============================================================
    # CHAT INPUT
    # ============================================================

    prompt = st.chat_input("Ask ShipNChat... or just say brew ☕")

    if prompt:

        if not api_key:
            st.error("🔑 Please set GROQ_API_KEY or OPENAI_API_KEY in your .env file")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # ── BREW EASTER EGG ──────────────────────────────────────
        if is_brew(prompt):
            brew_response = '<span class="brew-reply">freshly brewed</span> ☕'
            st.session_state.messages.append({"role": "assistant", "content": brew_response})
            with st.chat_message("assistant", avatar="☕"):
                st.markdown(brew_response, unsafe_allow_html=True)
            scroll_chat_to_bottom()

        # ── NORMAL RAG RESPONSE ───────────────────────────────────
        else:
            start_retrieve = time.time()
            bypass_active = st.session_state.get("bypass_rag", False)
            
            if st.session_state.uploaded_filename is not None:
                assert st.session_state.vectorstore is not None
                
                # Step 6: Test retrieval independently
                try:
                    q_emb = embedding_model.encode(["What is this document about?"])
                    test_results = st.session_state.vectorstore.query(
                        query_embeddings=q_emb.tolist(),
                        n_results=1,
                    )
                    print("[DEBUG TEST RETRIEVAL] Independent test query successful. Results:", test_results)
                except Exception as e:
                    print("[DEBUG TEST RETRIEVAL] Independent test query failed:", e)

                if bypass_active and st.session_state.chunks:
                    # Step 9: Bypass RAG and pass first 4 chunks directly
                    retrieval_result = {
                        "chunks": st.session_state.chunks[:4],
                        "scores": [1.0] * min(4, len(st.session_state.chunks)),
                        "best_score": 1.0,
                        "mean_score": 1.0,
                        "use_context": True,
                        "previews": [c[:300] + "..." for c in st.session_state.chunks[:4]]
                    }
                else:
                    retrieval_result = retrieve_relevant_chunks(prompt)
            else:
                retrieval_result = None
            retrieve_time = time.time() - start_retrieve

            st.session_state.diagnostics["retrieve_time"] = retrieve_time
            st.session_state.diagnostics["bypass_active"] = bypass_active
            
            if retrieval_result:
                st.session_state.diagnostics["retrieved_chunks_count"] = len(retrieval_result.get("chunks", []))
                st.session_state.diagnostics["similarity_scores"] = retrieval_result.get("scores", [])
                st.session_state.diagnostics["context_length"] = sum(len(c) for c in retrieval_result.get("chunks", []))
                st.session_state.diagnostics["retrieved_previews"] = retrieval_result.get("previews", [])

                print(f"[DEBUG RAG] Query: {prompt}")
                print(f"[DEBUG RAG] Chunks retrieved: {len(retrieval_result.get('chunks', []))}")
                print(f"[DEBUG RAG] Similarity scores: {retrieval_result.get('scores', [])}")
                print(f"[DEBUG RAG] Context length: {sum(len(c) for c in retrieval_result.get('chunks', []))} chars")

            # Check if PDF is uploaded but no chunks returned
            if st.session_state.uploaded_filename is not None and (not retrieval_result or not retrieval_result.get("chunks")):
                response = "PDF loaded, but no relevant context was found."
                st.session_state.diagnostics["llm_time"] = 0.0
            else:
                start_llm = time.time()
                with st.spinner("☕ ShipNChat is brewing your answer..."):
                    client   = build_client(api_key)
                    response = generate_response(client, model, prompt, retrieval_result)
                llm_time = time.time() - start_llm
                st.session_state.diagnostics["llm_time"] = llm_time

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant", avatar="☕"):
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
            if st.session_state.embeddings is not None:
                embeddings = st.session_state.embeddings[:30]
            else:
                embeddings = create_embeddings(st.session_state.chunks[:30])
            render_semantic_clustering(st.session_state.chunks[:30], embeddings)
    else:
        st.info("Upload a PDF in the sidebar to unlock the document intelligence dashboard.")
