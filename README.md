# Backend
Rag based Slm solution


# Step 1.
Install all the required dependencies:

* ChromaDB -
  pip install chromadb

* Sentence-Transformers -
  pip install -U sentence-transformers

* Torch -
pip install torch

* Stemmer -
pip install PyStemmer

* Transformers -
pip install transformers

* Fitz/pymupdf -
pip install pymupdf

* Groq model used with API token calls -
pip install groq


# Step 2.
Create "pdfs" folder in Backend root: location = Backend/pdfs
Add cleaned_pages.json file to pdfs folder *This file is a secret*


# Step 3.
Create  "Storage" folder in Backend root: location = Backend/Storage
Create "previous_sessions" folder in Storage: location = Backend/Storage/previous_sessions


# Step 4.
run -->
spell_parser.py
rule_parser.py
session_chunking.py
rule_chunking.py
dataStorage.py


# Step 5.
Go to api_chat.py and fill in your Groq key


# Step 6.
api_chat.py should now be working





* Something -
pip install gaming

