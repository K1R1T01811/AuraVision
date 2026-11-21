import time
from typing import List, Dict, Set
from collections import defaultdict


class AccumulatorState:
    def __init__(self, stable_objects, new_classes, all_confidences):
        self.stable_objects = stable_objects
        self.new_classes = new_classes
        self.all_confidences = all_confidences


class TemporalConfidenceAccumulator:
    def __init__(self, min_frames=5, decay_rate=0.85, threshold=0.7):
        self.min_frames = min_frames
        self.decay_rate = decay_rate
        self.threshold = threshold

        self.confidences = defaultdict(float)
        self.last_seen_classes: Set[str] = set()

    def update(self, detections) -> AccumulatorState:
        detected_classes = set(d.label for d in detections)

        # Scene-level novelty (NOT lifetime)
        new_classes = list(detected_classes - self.last_seen_classes)
        self.last_seen_classes = detected_classes.copy()

        # Update confidence
        for cls in list(self.confidences.keys()):
            if cls in detected_classes:
                self.confidences[cls] = min(
                    1.0, self.confidences[cls] + 1.0 / self.min_frames
                )
            else:
                self.confidences[cls] *= self.decay_rate
                if self.confidences[cls] < 0.01:
                    del self.confidences[cls]

        for cls in detected_classes:
            if cls not in self.confidences:
                self.confidences[cls] = 1.0 / self.min_frames

        stable_objects = [
            cls for cls, conf in self.confidences.items()
            if conf >= self.threshold
        ]

        return AccumulatorState(
            stable_objects=sorted(stable_objects),
            new_classes=sorted(new_classes),
            all_confidences=dict(self.confidences)
        )

    def reset(self):
        self.confidences.clear()
        self.last_seen_classes.clear()
