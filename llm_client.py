import os
from groq import Groq
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found in .env file!")

    return Groq(api_key=api_key)

def get_llm_response(client, question, context):
    prompt = f"""
    You are a Game Design Knowledge Assistant.
    Use ONLY the provided context to answer the question.
    If the context does not contain the answer, reply with:
    "I don't have enough information to answer that."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content