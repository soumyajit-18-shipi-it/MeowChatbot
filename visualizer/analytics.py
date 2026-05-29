import pandas as pd
from collections import Counter

from nlp.tokenizer import tokenize_text
from nlp.sentiment import analyze_sentiment
from visualizer.charts import (
    create_frequency_chart,
    create_cluster_chart
)

def build_analytics(text):

    tokens = tokenize_text(text)

    frequencies = Counter(tokens)

    common_words = frequencies.most_common(20)

    df = pd.DataFrame(
        common_words,
        columns=["Word", "Frequency"]
    )

    sentiment = analyze_sentiment(text)

    return {
        "frequency_df": df,
        "sentiment": sentiment
    }