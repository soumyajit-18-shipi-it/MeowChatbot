# ============================================================
# core/cache.py
# ============================================================

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb

# ============================================================
# EMBEDDING MODEL CACHE
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model

# ============================================================
# SENTIMENT MODEL CACHE
# ============================================================

@st.cache_resource
def load_sentiment_pipeline():

    sentiment_model = pipeline(
        "sentiment-analysis"
    )

    return sentiment_model

# ============================================================
# CHROMADB CACHE
# ============================================================

@st.cache_resource
def get_vector_database():

    client = chromadb.PersistentClient(
        path="./data/chroma_db"
    )

    collection = client.get_or_create_collection(
        name="shipnchat_collection"
    )

    return collection

# ============================================================
# PDF TEXT CACHE
# ============================================================

@st.cache_data
def cache_pdf_text(text):

    return text

# ============================================================
# CHUNK CACHE
# ============================================================

@st.cache_data
def cache_chunks(chunks):

    return chunks

# ============================================================
# EMBEDDINGS CACHE
# ============================================================

@st.cache_data
def cache_embeddings(_texts):

    model = load_embedding_model()

    embeddings = model.encode(_texts)

    return embeddings