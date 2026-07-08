import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    max_tokens=100,
    messages=[{"role": "user", "content": "say hello"}]
)
print(response.choices[0].message.content)