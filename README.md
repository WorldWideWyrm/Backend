# Backend
Rag based SLM solution


# Step 1.
Install all the required dependencies:

* **ChromaDB** -
  pip install chromadb

* **Sentence-Transformers** -
  pip install -U sentence-transformers

* **Torch** -
pip install torch

* **Stemmer** -
pip install PyStemmer

* **Transformers** -
pip install transformers

* **Fitz/pymupdf** -
pip install pymupdf

* **Groq model used with API token calls** -
pip install groq


# Step 2.
Create "pdfs" folder in Backend root: <br /> 
**Backend/** <br />

Add cleaned_pages.json file to pdfs folder *This file is a secret*: <br />
**Backend/pdfs** <br />


# Step 3.
Create  "Storage" folder outside of Backend root: <br /> 
**../Storage/** <br />

Create "previous_sessions" folder in Storage: <br /> 
**../Storage/previous_sessions** <br />


# Step 4.
Run the stated files below: <br />

**Backend/Parsers/:** <br />
spell_parser.py <br />
rule_parser.py <br />

**Backend/Chunking/:** <br />
session_chunking.py <br />
rule_chunking.py <br />

**Backend/chroma_storage/:** <br />
dataStorage.py <br />


# Step 5.
Go to api_chat.py and fill in your Groq key


# Step 6.
api_chat.py should now be working





* Something -
pip install gaming

