import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

# Set your Gemini API key (recommended: set as environment variable)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in your system/environment

if GEMINI_API_KEY is None:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)
for m in genai.list_models():
    print(m.name)