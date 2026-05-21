from groq import Groq
# client = Groq(  
#     api_key="")

SYSTEM_PROMPT = """
You are a helpful assistant answering questions on D&D 5.5e.

Rules:
- Do not invent sources or facts.
- If you do not now asnwer such
- Answer clearly and concisely.
"""

def chatCall(query, client):

    user_content = f"""
    User question:
    {query}
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    for i in range(500):
        query = input("What would you like to know? ").strip()
        if query == None:
            break
        print(chatCall(query))
