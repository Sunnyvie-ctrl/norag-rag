import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = os.getenv("LLM_MODEL", "gemini-3.6-flash")
print(f"Testing connection to Google using model: {model}...")

try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'Connection successful!'"}]
    )
    print("\nSUCCESS! The model replied:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"\nERROR: {str(e)}")