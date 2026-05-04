import sys
sys.path.append("Backend")

import api_chat
import json
import time
from pathlib import Path

INPUT_FILE = Path("qa_pairs.txt")
OUTPUT_FILE = Path("questions2.json")
ANSWERS_FILE = Path("answers2.json")

MAX_RETRIES = 5
BASE_WAIT_SECONDS = 60*30


def load_json(path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


questions = load_json(OUTPUT_FILE, [])
answers = load_json(ANSWERS_FILE, [])

# Resume from where we left off
completed_count = len(answers)

all_questions = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("Q:"):
            all_questions.append(line[2:].strip())

for index, question in enumerate(all_questions[completed_count:], start=completed_count):
    print(f"Processing {index + 1}/{len(all_questions)}")

    for attempt in range(MAX_RETRIES):
        try:
            answer, user_context = api_chat.chatCall(question)

            answers.append(answer)
            questions.append(user_context)

            # Save progress immediately after each success
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)

            with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
                json.dump(answers, f, indent=2, ensure_ascii=False)

            break

        except Exception as e:
            wait_time = BASE_WAIT_SECONDS * (2 ** attempt)

            print(f"Error on question {index + 1}: {e}")

            if attempt == MAX_RETRIES - 1:
                print("Max retries reached. Saving progress and stopping.")
                sys.exit(1)

            print(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)