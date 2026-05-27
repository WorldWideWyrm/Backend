from groq import Groq
from model_interface import BaseSTTStrategy

class GroqLlamaStrategy(BaseSTTStrategy):
    def __init__(self, key):
       self.client = Groq(api_key=key)
       self.SYSTEM_PROMPT = """
            You are a helpful assistant answering questions using the provided RAG context.

            Rules:
            - Use the RAG context over general knowledge.
            - Use the last session for if they as for notes or resumes from last session or time, it will only be included if the query uses the word 'last time', 'previous' or 'last session' is used.
            - If the RAG context or the last session does not contain the answer, say you do not know and give recomendation based on the RAG data and your own knowledge on how to rephrase the question.
            - If the question refers to all or each of something, the RAG will probably not return all of said category, then responed with examples from the RAG if possible but refer them to the book
            - Do not invent sources or facts.
            - Answer clearly and concisely.
            """

    def chat(self, messeage):
        chat_completion = self.client.chat.completions.create(
        messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": messeage},
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content