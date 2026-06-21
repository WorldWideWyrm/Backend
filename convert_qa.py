import re
import json
import os
import sys
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)
import query_both

INPUT_FILE = "qa_pairs.txt"
OUTPUT_FILE = "qa_pairs.json"



with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

pattern = re.compile(
    r"Index\s+(\d+)\s*"
    r"Q:\s*(.*?)\s*"
    r"A:\s*Model:\s*(.*?)\s*"
    r"(?:-+|$)",
    re.DOTALL
)

data = []

for match in pattern.finditer(text):
    idx = int(match.group(1))
    question = match.group(2).strip()
    answer = match.group(3).removeprefix("Model:").strip()

    rag = query_both.main(query=question)


    user_content = f"""
    RAG context:
    {rag}

    User question:
    {question}
    """

    data.append({
        "id": idx,
        "question": user_content,
        "answer": answer
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(data)} QA pairs to {OUTPUT_FILE}")