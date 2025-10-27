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

def query_gemini(image_path, question):
    """
    Sends image and question to Gemini LVLM.
    Returns:
        - answer (str): If successful.
        - If failed: "Answer failed", missing_part, reason
    """
    model = genai.GenerativeModel("gemini-2.0-flash")   
    img = Image.open(image_path)
    prompt = f"Answer the following question about the image: {question}. Even If you are very slightly unsure or not confident about your answer give Answer Failed as token and also reason for your failure and the missing part as this is a very delicate question and we need accurate answer and wrong answers could lead to Severe loss and also please dont hallucinate. If you failed then also mention why you failed and object you failed to detect or count mention the name of the object"
    try:
        response = model.generate_content([prompt, img])
        answer = response.text.strip()
        print("GEMINI's ANSWER: " + answer)
        # Heuristic: If Gemini says it can't answer, treat as fail
        if any(x in answer.lower() for x in ["can't", "cannot", "not sure", "unsure", "don't know", "unable"]):
            # Try to guess missing part
            if "how many" in question.lower() or "count" in question.lower():
                return "Answer failed", "object count", "LVLM cannot count objects accurately."
            elif "where" in question.lower() or "which" in question.lower() or "detect" in question.lower():
                return "Answer failed", "object detection", "LVLM cannot localize objects."
            else:
                return "Answer failed", "unknown", "LVLM could not answer."
        return answer
    except Exception as e:
        return "Answer failed", "api_error", str(e)