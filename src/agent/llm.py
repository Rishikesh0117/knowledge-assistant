from openai import OpenAI
from src import config

client = OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=config.NEBIUS_API_KEY)


def chat(messages, model=None, temperature=0.2, max_tokens=800):
    resp = client.chat.completions.create(
        model=model or config.FAST_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,  # keeps every call cheap
    )
    return resp.choices[0].message.content


def embed(texts, model=None):
    resp = client.embeddings.create(model=model or config.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Say hello in one short sentence."}]))