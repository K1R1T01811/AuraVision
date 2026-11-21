from ultralytics import YOLO
import numpy as np
import os
from typing import List, Tuple
from collections import defaultdict


def verify_model_path(path: str) -> None:
    """
    Verify that a YOLO model path exists and is valid.
    Fail fast with clear error messages.
    """
    if not isinstance(path, str):
        raise TypeError(f"Model path must be a string, got {type(path)}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ YOLO model not found at path: {path}")

    if not path.lower().endswith(".pt"):
        raise ValueError(f"❌ Invalid YOLO model file (expected .pt): {path}")


class Detection:
    """Represents a single object detection from YOLO"""
    def __init__(self, label: str, confidence: float, bbox: Tuple[float, float, float, float]):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)

    def __repr__(self):
        return f"Detection(label={self.label}, conf={self.confidence:.2f})"


class DetectionFusion:
    """
    Merges detections from multiple YOLO models using NMS-based deduplication.
    Prevents duplicate detections from overlapping bounding boxes.
    """
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def merge(self, detections_a: List[Detection], detections_b: List[Detection]) -> List[Detection]:
        all_detections = detections_a + detections_b
        if not all_detections:
            return []

        by_label = defaultdict(list)
        for det in all_detections:
            by_label[det.label].append(det)

        merged = []
        for label, dets in by_label.items():
            merged.extend(self._apply_nms(dets))

        return merged

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)
            sorted_dets = [
                det for det in sorted_dets
                if self._compute_iou(best.bbox, det.bbox) < self.iou_threshold
            ]

        return keep

    def _compute_iou(self, box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0


class DualYOLOSentinel:
    """
    Dual YOLO detector that runs:
    - Model A: general COCO YOLO
    - Model B: custom-trained YOLO (optional)

    Merges detections into a unified stream.
    Backward compatible with single YOLO mode.
    """

    def __init__(
        self,
        general_model_path="yolov8n.pt",
        custom_model_path=None,
        confidence_threshold=0.25,
        fusion_iou_threshold=0.5
    ):
        print("🔍 Initializing YOLO Sentinel")

        # --- General YOLO ---
        print(f"   Loading general model: {general_model_path}")
        verify_model_path(general_model_path)
        self.model_a = YOLO(general_model_path)
        self.confidence_threshold = confidence_threshold

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model_a(dummy, verbose=False)

        print("   ✅ General YOLO ready")
        print("   Classes:", list(self.model_a.names.values())[:10], "...")

        # --- Custom YOLO (optional) ---
        self.model_b = None
        if custom_model_path:
            try:
                print(f"   Loading custom model: {custom_model_path}")
                verify_model_path(custom_model_path)
                self.model_b = YOLO(custom_model_path)
                _ = self.model_b(dummy, verbose=False)
                print("   ✅ Custom YOLO ready")
                print("   Custom classes:", list(self.model_b.names.values()))
            except Exception as e:
                print(f"   ⚠️ Failed to load custom YOLO: {e}")
                print("   → Falling back to single YOLO mode")
                self.model_b = None

        self.fusion = DetectionFusion(iou_threshold=fusion_iou_threshold)

        if self.model_b:
            print("🎯 Dual YOLO mode ENABLED")
        else:
            print("🎯 Single YOLO mode (backward compatible)")

    def detect(self, frame) -> List[Detection]:
        detections_a = self._run_yolo(self.model_a, frame)

        if self.model_b is not None:
            detections_b = self._run_yolo(self.model_b, frame)
            return self.fusion.merge(detections_a, detections_b)

        return detections_a

    def _run_yolo(self, model, frame) -> List[Detection]:
        results = model(frame, verbose=False, conf=self.confidence_threshold)
        detections = []

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy()
                label = model.names[cls_id]

                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=tuple(bbox)
                    )
                )

        return detections

    def get_class_names(self, detections: List[Detection]) -> List[str]:
        return sorted(set(d.label for d in detections))


# 🔁 Backward compatibility
YOLOSentinel = DualYOLOSentinel
