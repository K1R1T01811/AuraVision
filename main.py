import cv2
import time
import threading

from camera.webcam import Webcam
from vision.sentinel import DualYOLOSentinel
from vision.temporal_confidence import TemporalConfidenceAccumulator
from vision.semantic_trigger import SemanticTrigger
from vision.captioning import BLIPCaptioner
from ai.reasoning import generate_speech, FALLBACK_SENTENCE
from audio.tts import speak
from utils.temporal_scene_memory import TemporalSceneMemory
from utils.semantic_context import SemanticContext

# ---------------- STARTUP ----------------
print("🔊 AI Vision Assistant Started (Option C: Dual YOLO + BLIP)")

frame_id = 0

# ---------------- CONFIG ----------------
WINDOW_NAME = "AI Vision Assistant"
YOLO_MODEL = "yolov8n.pt"
CUSTOM_YOLO_MODEL = "weights/yolo_custom.pt"

MIN_FRAMES_FOR_CONFIDENCE = 5
CONFIDENCE_THRESHOLD = 0.7
MIN_TRIGGER_INTERVAL = 2.0
SCENE_MEMORY_TTL = 8.0
ASSURANCE_TIMEOUT = 30.0
# ----------------------------------------

# Shared state
latest_frame = None
latest_frame_id = 0
latest_detections = []

frame_lock = threading.Lock()
detections_lock = threading.Lock()
vision_busy = False

# ---------------- INIT ----------------
camera = Webcam()

sentinel = DualYOLOSentinel(
    general_model_path=YOLO_MODEL,
    custom_model_path=CUSTOM_YOLO_MODEL,
    confidence_threshold=0.25
)

accumulator = TemporalConfidenceAccumulator(
    min_frames=MIN_FRAMES_FOR_CONFIDENCE,
    decay_rate=0.85,
    threshold=CONFIDENCE_THRESHOLD
)

trigger = SemanticTrigger(min_trigger_interval=MIN_TRIGGER_INTERVAL)
blip = BLIPCaptioner()
scene_memory = TemporalSceneMemory(ttl=SCENE_MEMORY_TTL)
semantic_context = SemanticContext(ttl=SCENE_MEMORY_TTL)

# ---------------- SPEECH ----------------
def generate_and_speak(speech_type, frame_id, **kwargs):
    try:
        output = generate_speech(speech_type=speech_type, **kwargs)
        if output:
            print(f"[Frame {frame_id}] 🗣️ {output}")
            speak(output)
            semantic_context.mark_speech_event(
                is_assurance=(speech_type == "assurance")
            )
            return
    except Exception as e:
        print("❌ Speech error:", e)

    print(f"[Frame {frame_id}] 🗣️ FALLBACK")
    speak(FALLBACK_SENTENCE)
    semantic_context.mark_speech_event(
        is_assurance=(speech_type == "assurance")
    )

# ---------------- SEMANTIC WORKER ----------------
def semantic_worker():
    global vision_busy

    while True:
        time.sleep(0.05)

        with frame_lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()
            fid = latest_frame_id

        with detections_lock:
            detections_copy = latest_detections.copy()

        acc_state = accumulator.update(detections_copy)

        # 🔔 Assurance
        people = acc_state.stable_objects.count("person")
        if semantic_context.should_trigger_assurance(people, ASSURANCE_TIMEOUT):
            if not vision_busy:
                vision_busy = True
                generate_and_speak(
                    speech_type="assurance",
                    frame_id=fid,
                    people_count=people
                )
                vision_busy = False

        should_trigger, reason = trigger.should_trigger(acc_state)
        if not should_trigger or vision_busy:
            continue

        vision_busy = True
        print(f"🎯 Trigger: {reason}")

        try:
            caption = blip.caption(frame_copy)
            objects = acc_state.stable_objects

            if not caption and not objects:
                continue

            cache_key = {
                "objects": tuple(objects),
                "caption": caption
            }

            if scene_memory.is_duplicate(cache_key):
                print("⏭️ Duplicate scene, skipping")
                continue

            scene_memory.update(cache_key)

            generate_and_speak(
                speech_type="event",
                frame_id=fid,
                objects=objects,
                caption=caption,
                trigger_reason=reason
            )

        finally:
            vision_busy = False

# ---------------- THREAD ----------------
threading.Thread(target=semantic_worker, daemon=True).start()

# ---------------- MAIN LOOP ----------------
print("📹 Starting sentinel loop (Option C)")
print("Press 'q' to exit")

while True:
    frame = camera.read()
    if frame is None:
        continue

    frame_id += 1

    detections = sentinel.detect(frame)

    with frame_lock:
        latest_frame = frame.copy()
        latest_frame_id = frame_id

    with detections_lock:
        latest_detections = detections

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print("👋 Vision Assistant stopped")
