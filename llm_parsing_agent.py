from agents.clip_count_agent import ClipCountAgent
from agents.grounded_sam_agent import GroundedSAMAgent
from agents.ocr_agent import OcrAgent
import cv2

def llm_parsing_agent(image_path, question, missing_part):
    if "count" in missing_part:
        # Use ClipCountAgent
        agent = ClipCountAgent("tools/clip/CLIP-Count/weights/clipcount_pretrained.ckpt", device="cpu")
        image_np = cv2.imread(image_path)
        # Try to extract object name from question
        needed_object = question.lower().replace("how many", "").replace("are there", "").strip().split()[0]
        count, _ = agent.detect_count(image_np, needed_object)
        return f"Counted {count} {needed_object}(s)."
    elif "detection" in missing_part or "localize" in missing_part:
        # Use GroundedSAMAgent
        agent = GroundedSAMAgent(
            grounding_dino_config="tools/grounded-sam/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounding_dino_ckpt="tools/grounded-sam/Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth",
            sam_encoder_version="vit_h",
            sam_ckpt="tools/grounded-sam/Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth"
        )
        detections, det_img = agent.detect(image_path, [question])
        return f"Detected from GroundingSAMagent: {detections}"
    elif "text" in missing_part or "read" in missing_part:
        agent = OcrAgent()
        extracted_text = agent.extract_text(image_path)
        return f"Extracted Text from OCR agent: {extracted_text}"
    else:
        return "Unable to parse missing part."