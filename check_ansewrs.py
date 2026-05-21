import json
from pathlib import Path

QUESTION_FILE = Path("questions.json")
ANSWERS_RAG_FILE = Path("answers-rag.json")
ANSWERS_NO_RAG_FILE = Path("answers_no_rag.json")
ANSWERS_NO_RAG_NO_DETAILS_FILE = Path("answers_no_rag_no_details.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


questions = load_json(QUESTION_FILE)
answers_rag = load_json(ANSWERS_RAG_FILE)
answers_no_rag = load_json(ANSWERS_NO_RAG_FILE)
answers_no_rag_no_details = load_json(ANSWERS_NO_RAG_NO_DETAILS_FILE)

total = min(
    len(questions),
    len(answers_rag),
    len(answers_no_rag),
    len(answers_no_rag_no_details)
)

for i in range(total):
    print("\n" + "=" * 100)
    print(f"QUESTION {i + 1}/{total}")
    print("=" * 100)

    print("\nQUESTION:")
    print(questions[i])

    print("\n--- RAG ---")
    print(answers_rag[i])

    print("\n--- NO RAG ---")
    print(answers_no_rag[i])

    print("\n--- NO RAG NO DETAILS ---")
    print(answers_no_rag_no_details[i])

    input("\nPress ENTER to continue...")