# ============================================================
# ENHANCED ANALYTICS MODULE
# Advanced AI Document Intelligence System
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import textstat
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# KEYWORD EXTRACTION & NLP ANALYSIS
# ============================================================

def extract_advanced_keywords(text, embedding_model, top_n=15):
    """Extract keywords using multiple methods"""
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT(model=embedding_model)
        keywords = kw_model.extract_keywords(
            text[:5000],
            language="english",
            top_n=top_n,
            use_mmr=True,
            diversity=0.5
        )
        return [kw[0] for kw in keywords]
    except Exception as e:
        st.warning(f"Keyword extraction failed: {e}")
        from nltk import word_tokenize
        from nltk.corpus import stopwords
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [t for t in tokens if len(t) > 4 and t not in stop_words]
        return list(set(keywords))[:top_n]


def analyze_reading_complexity(text):
    """Calculate reading difficulty metrics"""
    try:
        flesch_score = textstat.flesch_reading_ease(text)
        complexity_level = "Easy" if flesch_score > 60 else "Medium" if flesch_score > 40 else "Complex"
        reading_time = max(1, len(text.split()) // 200)
        
        return {
            "flesch_score": round(flesch_score, 1),
            "complexity": complexity_level,
            "reading_time": f"{reading_time} min",
            "avg_sentence_length": round(textstat.avg_sentence_length(text), 1)
        }
    except Exception as e:
        return {"error": str(e)}


def extract_entities(text):
    """Extract named entities (organizations, persons, locations)"""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            st.info("Downloading spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        
        doc = nlp(text[:3000])
        
        entities = {
            "ORG": [],
            "PERSON": [],
            "GPE": [],
            "PRODUCT": [],
            "EVENT": []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        # Remove duplicates and limit
        for key in entities:
            entities[key] = list(set(entities[key]))[:5]
        
        return entities
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# SENTIMENT ANALYSIS - ADVANCED
# ============================================================

def analyze_emotional_tone(text, sentiment_pipeline):
    """Analyze document emotional tone"""
    try:
        if not sentiment_pipeline:
            return None
        
        sentiment = sentiment_pipeline(text[:512])[0]
        
        # Map to emotion categories
        emotion_map = {
            "POSITIVE": {"label": "Positive", "emotion": "Optimistic", "color": "#10b981"},
            "NEGATIVE": {"label": "Negative", "emotion": "Critical", "color": "#ef4444"},
            "NEUTRAL": {"label": "Neutral", "emotion": "Analytical", "color": "#6b7280"}
        }
        
        return {
            "label": sentiment["label"],
            "confidence": round(sentiment["score"] * 100, 1),
            "emotion": emotion_map.get(sentiment["label"], {}).get("emotion", "Unknown"),
            "color": emotion_map.get(sentiment["label"], {}).get("color", "#999")
        }
    except Exception as e:
        return None


def detect_document_tone(text):
    """Detect overall writing tone"""
    try:
        sentences = text.split('.')[:20]
        tone_indicators = {
            "urgent": len([s for s in sentences if any(w in s.lower() for w in ["urgent", "critical", "emergency", "immediately"])]),
            "technical": len([s for s in sentences if any(w in s.lower() for w in ["algorithm", "database", "system", "implementation"])]),
            "formal": len([s for s in sentences if any(w in s.lower() for w in ["therefore", "furthermore", "conclusion", "analysis"])]),
        }
        
        max_tone = max(tone_indicators.items(), key=lambda x: x[1])
        return {
            "primary_tone": max_tone[0].title(),
            "indicators": tone_indicators,
            "formality_score": round((tone_indicators["formal"] / len(sentences)) * 100, 1)
        }
    except:
        return {"primary_tone": "Neutral", "formality_score": 50}


# ============================================================
# CLUSTERING - ADVANCED
# ============================================================

def create_semantic_map(embeddings, chunks, n_clusters=5):
    """Create semantic map with clustering"""
    try:
        from sklearn.cluster import KMeans
        
        if len(embeddings) < 3:
            return None
        
        # Reduce dimensions for visualization
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        pca = PCA(n_components=min(3, len(embeddings)-1))
        reduced = pca.fit_transform(embeddings_scaled)
        
        # Clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
        labels = kmeans.fit_predict(embeddings_scaled)
        
        # Create cluster labels
        cluster_names = generate_cluster_names(chunks, labels)
        
        df = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1] if reduced.shape[1] > 1 else 0,
            "cluster": labels,
            "chunk_id": range(len(chunks)),
            "cluster_name": [cluster_names.get(l, f"Cluster {l}") for l in labels],
            "text_preview": [c[:60] + "..." if len(c) > 60 else c for c in chunks]
        })
        
        return df
    except Exception as e:
        st.error(f"Clustering error: {e}")
        return None


def generate_cluster_names(chunks, labels):
    """Generate meaningful cluster names"""
    cluster_names = {}
    
    for cluster_id in set(labels):
        cluster_chunks = [chunks[i] for i, l in enumerate(labels) if l == cluster_id]
        merged_text = " ".join(cluster_chunks[:3])
        
        # Simple heuristic naming
        if any(word in merged_text.lower() for word in ["introduction", "overview", "background"]):
            cluster_names[cluster_id] = "Introduction & Background"
        elif any(word in merged_text.lower() for word in ["method", "implementation", "process", "algorithm"]):
            cluster_names[cluster_id] = "Implementation & Methods"
        elif any(word in merged_text.lower() for word in ["result", "conclusion", "summary", "finding"]):
            cluster_names[cluster_id] = "Results & Conclusions"
        elif any(word in merged_text.lower() for word in ["technical", "code", "system", "architecture"]):
            cluster_names[cluster_id] = "Technical Details"
        else:
            cluster_names[cluster_id] = f"Topic {cluster_id + 1}"
    
    return cluster_names


# ============================================================
# AI INSIGHTS PANEL
# ============================================================

def generate_ai_insights(text, embedding_model, sentiment_pipeline):
    """Generate AI insights about the document"""
    try:
        insights = {}
        
        # Document category
        category_keywords = {
            "Technical": ["algorithm", "code", "implementation", "architecture", "system"],
            "Business": ["revenue", "market", "strategy", "competition", "growth"],
            "Academic": ["research", "study", "methodology", "analysis", "hypothesis"],
            "Financial": ["revenue", "earnings", "profit", "balance sheet", "financial"],
            "Medical": ["patient", "treatment", "disease", "clinical", "diagnosis"],
        }
        
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in category_keywords.items():
            score = sum(text_lower.count(kw) for kw in keywords)
            category_scores[category] = score
        
        insights["category"] = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else "General"
        
        # Complexity analysis
        complexity = analyze_reading_complexity(text)
        insights["complexity"] = complexity.get("complexity", "Unknown")
        
        # Tone detection
        tone = detect_document_tone(text)
        insights["writing_tone"] = tone.get("primary_tone", "Neutral")
        
        # Main themes
        keywords = extract_advanced_keywords(text, embedding_model, top_n=5)
        insights["main_themes"] = keywords
        
        # Sentiment
        sentiment = analyze_emotional_tone(text, sentiment_pipeline)
        if sentiment:
            insights["emotional_tone"] = sentiment["emotion"]
        
        return insights
    except Exception as e:
        return {"error": str(e)}


def format_insights_text(insights):
    """Format insights for display"""
    try:
        if "error" in insights:
            return f"Analysis unavailable: {insights['error']}"
        
        themes = ", ".join(insights.get("main_themes", []))
        category = insights.get("category", "General")
        complexity = insights.get("complexity", "Unknown")
        tone = insights.get("writing_tone", "Neutral")
        emotion = insights.get("emotional_tone", "Neutral")
        
        return f"""
        **Document Profile:**
        - **Category**: {category}
        - **Writing Tone**: {tone}
        - **Emotional Tone**: {emotion}
        - **Complexity Level**: {complexity}
        - **Main Themes**: {themes}
        """
    except:
        return "Unable to generate insights"


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_keyword_cloud_data(keywords, freq):
    """Prepare data for keyword visualization"""
    df = pd.DataFrame({
        "keyword": keywords,
        "frequency": freq
    }).sort_values("frequency", ascending=False)
    
    return df


def plot_advanced_keywords(keywords, frequencies):
    """Create interactive keyword visualization"""
    df = pd.DataFrame({
        "Keyword": keywords,
        "Frequency": frequencies
    }).sort_values("Frequency", ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            y=df["Keyword"],
            x=df["Frequency"],
            orientation='h',
            marker=dict(
                color=df["Frequency"],
                colorscale="Viridis",
                showscale=False
            ),
            hovertemplate="<b>%{y}</b><br>Frequency: %{x}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Key Topics & Terms",
        xaxis_title="Frequency",
        yaxis_title="Keywords",
        height=400,
        showlegend=False,
        hovermode='closest',
        template="plotly_dark"
    )
    
    return fig


def plot_semantic_map(df):
    """Create interactive semantic clustering visualization"""
    if df is None or len(df) == 0:
        return None
    
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_name",
        hover_data={"x": ":.2f", "y": ":.2f", "text_preview": True},
        title="Semantic Document Map",
        labels={"x": "Semantic Dimension 1", "y": "Semantic Dimension 2"},
        color_discrete_sequence=colors,
        height=500
    )
    
    fig.update_traces(
        marker=dict(size=10, opacity=0.7),
        hovertemplate="<b>%{customdata[2]}</b><br>" +
                     "Content: %{customdata[3]}<br>" +
                     "<extra></extra>"
    )
    
    fig.update_layout(
        template="plotly_dark",
        hovermode="closest"
    )
    
    return fig


def plot_sentiment_gauge(sentiment_data):
    """Create sentiment gauge visualization"""
    if not sentiment_data:
        return None
    
    fig = go.Figure(data=[
        go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_data["confidence"],
            title={"text": f"Emotional Tone: {sentiment_data['emotion']}"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": sentiment_data["color"]},
                "steps": [
                    {"range": [0, 33], "color": "#fee2e2"},
                    {"range": [33, 66], "color": "#fef3c7"},
                    {"range": [66, 100], "color": "#dcfce7"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90
                }
            }
        )
    ])
    
    fig.update_layout(height=400, template="plotly_dark")
    return fig


def plot_document_complexity(complexity_data):
    """Visualize document complexity metrics"""
    if "error" in complexity_data:
        return None
    
    metrics = {
        "Readability\nScore": complexity_data["flesch_score"],
        "Sentence\nLength": min(complexity_data["avg_sentence_length"] * 5, 100)
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker=dict(
                color=["#3b82f6", "#8b5cf6"],
                opacity=0.7
            ),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Document Complexity Metrics",
        yaxis_title="Score (0-100)",
        height=300,
        showlegend=False,
        template="plotly_dark"
    )
    
    return fig
