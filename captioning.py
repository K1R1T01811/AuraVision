from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2


class BLIPCaptioner:
    def __init__(self):
        # 🔒 FORCE CPU + FP32
        self.device = torch.device("cpu")

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_safetensors=True
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float32,
            use_safetensors=True
        ).to(self.device)

        self.model.eval()

    def caption(self, frame):
        # OpenCV BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # 🔒 KEEP EVERYTHING ON CPU (FP32)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=25
            )

        caption = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return caption.strip()

    def caption_region(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            return ""

        return self.caption(cropped)

    def caption_regions(self, frame, bboxes):
        return [self.caption_region(frame, bbox) for bbox in bboxes]
