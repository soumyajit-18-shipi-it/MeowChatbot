import chromadb

from core.embeddings import generate_embeddings

client = chromadb.PersistentClient(
    path="./data/chroma_db"
)

collection = client.get_or_create_collection(
    name="meowbot_collection"
)

def store_documents(chunks):

    embeddings = generate_embeddings(chunks)

    ids = [f"doc_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids
    )

def retrieve_chunks(query, n_results=4):

    query_embedding = generate_embeddings([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )

    return results["documents"][0]