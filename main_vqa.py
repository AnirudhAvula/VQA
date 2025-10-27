from llm_integration import query_gemini
from llm_parsing_agent import llm_parsing_agent
from agents.ocr_agent import OcrAgent

def vqa_pipeline(image_path, question):
      # Directly use the OCR agent for testing
    print("Temporarily bypassing Gemini and testing OCR agent directly.")
    agent = OcrAgent()
    extracted_text = agent.extract_text(image_path)
    return f"Extracted Text: {extracted_text}"
    result = query_gemini(image_path, question)
    print("RESULT: ")
    print(result)
    if isinstance(result, tuple) and result[0] == "Answer failed":
        _, missing_part, reason = result
        print(f"LVLM failed: {reason}. Missing: {missing_part}")
        answer = llm_parsing_agent(image_path, question, missing_part)
        return answer
    else:
        return result

if __name__ == "__main__":
    image_path = "ocr3.jpg"  # Change to your image path
    question = "what is text written in this image?"
    answer = vqa_pipeline(image_path, question)
    print("Final Answer:", answer)