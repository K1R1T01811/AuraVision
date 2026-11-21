import torch
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection


class GroundingDINO:
    def __init__(self):
        # 🔒 HARD LOCK: CPU + FP32
        self.device = torch.device("cpu")
        torch.set_default_dtype(torch.float32)

        self.processor = GroundingDinoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        )

        self.model = GroundingDinoForObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(self.device)

        self.model.eval()

    def detect(self, image, text_prompt, score_threshold=0.35):
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        )

        # Force CPU FP32
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.autocast(device_type="cpu", enabled=False):
                outputs = self.model(**inputs)

        # ⚠️ Older transformers API (no keyword args)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            target_sizes=[image.shape[:2]]
        )[0]

        labels = results.get("text_labels", [])
        scores = results.get("scores", [])

        # ✅ Manual confidence filtering
        clean_labels = []
        for label, score in zip(labels, scores):
            if score >= score_threshold:
                clean_labels.append(label.strip())

        return sorted(set(clean_labels))

    def build_prompt_from_detections(self, detections):
        """
        Build a text prompt from YOLO detections for dynamic grounding.
        
        Args:
            detections: List of Detection objects from YOLOSentinel
            
        Returns:
            Comma-separated string of unique class names
        """
        if not detections:
            return "object, thing, item"  # Generic fallback
        
        unique_classes = sorted(set(d.label for d in detections))
        return ", ".join(unique_classes)

