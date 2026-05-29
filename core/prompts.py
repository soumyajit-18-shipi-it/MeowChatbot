from core.rag import retrieve_chunks

def build_prompt(user_input):

    chunks = retrieve_chunks(user_input)

    context = "\n\n".join(chunks)

    return f"""
You are MeowBot AI X.

Use the retrieved context to answer accurately.

CONTEXT:
{context}

QUESTION:
{user_input}
"""