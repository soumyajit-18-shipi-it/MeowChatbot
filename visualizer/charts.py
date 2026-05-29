import plotly.express as px

def create_frequency_chart(df):

    fig = px.bar(
        df,
        x="Word",
        y="Frequency",
        title="Word Frequency"
    )

    return fig

def create_cluster_chart(df):

    fig = px.scatter(
        df,
        x="Chunk",
        y="Cluster",
        title="Semantic Clustering"
    )

    return fig