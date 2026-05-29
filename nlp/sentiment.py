from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

pipeline_model = load_sentiment_pipeline()

def analyze_sentiment(text):

    result = pipeline_model(text[:512])

    return result[0]