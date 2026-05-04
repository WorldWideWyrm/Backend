import datetime
import os

from groq import Groq
import sys
sys.path.append("Backend")
sys.path.append("Chunking")
import query_both
import session_chunking

CLIENT = Groq(
)

SYSTEM_PROMPT = """
You are a helpful assistant answering questions using the provided RAG context.

Rules:
- Use the RAG context over general knowledge.
- If the RAG context does not contain the answer, say you do not know and give reccomendation based on the rag data and your own knoledge on how to refrase the question.
- Do not invent sources or facts.
- Answer clearly and concisely.
"""

def chatCall(query, new_session=None):

  
  session_chunking.update(new_session)
     
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

def get_last_session():
    parent_dir = os.path.dirname(os.getcwd())

    target_path = os.path.join(parent_dir, "front_end", "previous_sessions")

    file_texts = []
    today = datetime.date.today()
    
    for filename in sorted(os.listdir(target_path), key=session_chunking.session_key):
        file_path = os.path.join(target_path, filename)
        # Get file creation/modification time
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path)).date()

        # Skip files created/modified today
        if file_time == today:
          continue
        if os.path.isfile(file_path):  
            with open(file_path, "r", encoding="utf-8") as f:
                file_texts.append(f.read())
    return file_texts[len(file_texts)-1]

if __name__ == "__main__":
    query = input("What would you like to know? ").strip()
    print(chatCall(query))