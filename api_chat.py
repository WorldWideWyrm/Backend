import os

from groq import Groq
import sys
sys.path.append("Backend")
import query_both

CLIENT = Groq(
    api_key=#needs key here,
)

SYSTEM_PROMPT = """
You are a helpful assistant answering questions using the provided RAG context.

Rules:
- Use the RAG context over general knowledge.
- If the RAG context does not contain the answer, say you do not know and give reccomendation based on the rag data and your own knoledge on how to refrase the question.
- Do not invent sources or facts.
- Answer clearly and concisely.
"""

def chatCall(query):
  rag = query_both.main(query=query)

  chat_completion = CLIENT.chat.completions.create(
      messages=[
          {
                "role": "system",
                "content": SYSTEM_PROMPT,
          },
          {
              "role": "user",
              "content": f"""
                RAG context:
                {rag}

                User question:
                {query}
                """,
          }
      ],
      model="llama-3.3-70b-versatile",
  )
  return chat_completion.choices[0].message.content

if __name__ == "__main__":
    query = input("What would you like to know? ").strip()
    print(chatCall(query))