import cv2
from agents.grounded_sam_agent import GroundedSAMAgent

agent = GroundedSAMAgent(
    grounding_dino_config="tools/grounded-sam/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounding_dino_ckpt="tools/grounded-sam/Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth",
    sam_encoder_version="vit_h",
    sam_ckpt="tools/grounded-sam/Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth"
)

# Detection only
detections, det_img = agent.detect("demo2.jpg", ["The running dog"])
cv2.imwrite("output_detect_only.jpg", det_img)

# Full detect + segment
detections, seg_img = agent.detect_and_segment("demo2.jpg", ["The running dog"])
cv2.imwrite("output_detect_segment.jpg", seg_img)
