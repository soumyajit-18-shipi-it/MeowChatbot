from sklearn.cluster import KMeans
import pandas as pd

from core.embeddings import generate_embeddings

def cluster_documents(chunks):

    embeddings = generate_embeddings(chunks)

    model = KMeans(
        n_clusters=3,
        random_state=42
    )

    labels = model.fit_predict(embeddings)

    return pd.DataFrame({
        "Chunk": range(len(chunks)),
        "Cluster": labels
    })