import datetime
import os

from groq import Groq
import sys
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKING_DIR = os.path.join(BACKEND_DIR, "Chunking")

sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, CHUNKING_DIR)
import query_both
import session_chunking
import time

client = Groq(
    api_key=""

)

SYSTEM_PROMPT = """
You are a helpful assistant answering questions using the provided RAG context.

Rules:
- Use the RAG context over general knowledge.
- Use the last session for if they as for notes or resumes from last session or time, it will only be included it they use the word 'last time', 'previous' or 'last session' is used.
- If the RAG context or the last session does not contain the answer, say you do not know and give reccomendation based on the rag data and your own knoledge on how to refrase the question.
- Do not invent sources or facts.
- Answer clearly and concisely.
"""

def chatCall(query, new_session=None):
    session_chunking.update(new_session)
     
    rag = query_both.main(query=query)

    include_last = should_include_last_session(query)

    user_content = f"""
    RAG context:
    {rag}

    User question:
    {query}
    """

    if include_last:
        last_session = get_last_session()
        if(last_session!=None):
            user_content += f"\n\nLast session:\n{last_session}"
        else:
            user_content += f"\n\nLast session:\n No last session"

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content#, user_content 

def should_include_last_session(query):
    keywords = ["last time", "previous", "last session"]
    return any(k in query.lower() for k in keywords)

def get_last_session():
    parent_dir = os.path.dirname(os.getcwd())

    target_path = os.path.join(parent_dir, "front_end", "previous_sessions")

    file_texts = []
    today = datetime.date.today()
    
    for filename in sorted(os.listdir(target_path), key=session_chunking.session_key):
        file_path = os.path.join(target_path, filename)
        # Get file creation/modification time
        file_time = time.ctime(os.path.getctime(file_path))
        # Skip files created/modified today
        if file_time == today:
          continue
        if os.path.isfile(file_path):  
            with open(file_path, "r", encoding="utf-8") as f:
                file_texts.append(f.read())
    if len(file_texts) == 0:
        return None
    return file_texts[len(file_texts)-1]

if __name__ == "__main__":
    for i in range(500):
        query = input("What would you like to know? ").strip()
        if query == None:
            break
        print(chatCall(query))
