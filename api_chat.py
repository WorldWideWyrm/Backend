from pathlib import Path
import re

current = Path(__file__).resolve()

# walk upward through parent folders
for parent in current.parents:
    storage_path = parent / "Storage"

    if storage_path.exists() and storage_path.is_dir():
        STORAGE_DIR = storage_path
        break
else:
    raise FileNotFoundError("Could not find storage folder")

import datetime
import os

from groq_llama_strategy import GroqLlamaStrategy
import sys
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKING_DIR = os.path.join(BACKEND_DIR, "Chunking")

sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, CHUNKING_DIR)
import query_both
import session_chunking
import time

chat_strategy = GroqLlamaStrategy("") # api_key only string

def chatCall(query, new_session=None):
    session_chunking.update(new_session)

    re.sub(r'[^a-zA-Z0-9!?,. ]', '', query)
     
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

    
    return chat_strategy.chat(user_content)

def should_include_last_session(query):
    keywords = ["last time", "previous", "last session"]
    return any(k in query.lower() for k in keywords)

def get_last_session():
    target_path = os.path.join(storage_path, "previous_sessions")

    file_texts = []
    today = datetime.datetime.now().date()

    for filename in sorted(os.listdir(target_path), key=session_chunking.session_key):
        file_path = os.path.join(target_path, filename)
        # Get file creation/modification time
        file_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).date()
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
