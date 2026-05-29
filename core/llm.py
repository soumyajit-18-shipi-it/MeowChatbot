from openai import OpenAI

def build_client(api_key):

    if api_key.startswith("gsk_"):
        return OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    return OpenAI(api_key=api_key)

def generate_response(
    client,
    model,
    messages,
    temperature,
    max_tokens,
    top_p
):

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )

    return response.choices[0].message.content