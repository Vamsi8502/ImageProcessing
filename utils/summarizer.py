import os
from google.auth import load_credentials_from_file
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ✅ Point to your service account JSON
SERVICE_ACCOUNT_KEY_PATH = "/Users/vamsikrishna/Documents/LEARNING PROJECTS/AI PROJECTS/ImageProcessing/new_key.json"

# ✅ Load credentials with correct scope
credentials, _ = load_credentials_from_file(
    SERVICE_ACCOUNT_KEY_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# ✅ Configure genai with credentials
genai.configure(credentials=credentials)

# ✅ Initialize Gemini model
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash",
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

def summarize_text(claim_text: str) -> str:
    prompt = f"""
You are an expert insurance assistant.

Analyze the following:

{claim_text}

Return a summary in this format:

- 📝 Summary: <summary of the user claim>
- 🔍 Visual Label Relevance: <does the image content support or contradict the claim?>
"""
    response = model.generate_content(prompt)
    return response.text.strip()
