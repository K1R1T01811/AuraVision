import time
import threading


class SemanticContext:
    """
    Thread-safe semantic context manager.
    Tracks speech timing, captions, and grounded objects.
    """

    def __init__(self, ttl=8.0):
        self.ttl = ttl
        self.lock = threading.Lock()

        # Vision memory
        self.last_caption = None
        self.caption_timestamp = 0

        self.grounded_objects = []
        self.objects_timestamp = 0

        # Speech tracking
        self.last_spoken_timestamp = 0
        self.last_assurance_timestamp = 0

    # ------------------------
    # VISION STORAGE
    # ------------------------
    def update_caption(self, caption):
        with self.lock:
            self.last_caption = caption
            self.caption_timestamp = time.time()

    def update_objects(self, objects):
        with self.lock:
            self.grounded_objects = objects or []
            self.objects_timestamp = time.time()

    # ------------------------
    # SPEECH TRACKING (FIX)
    # ------------------------
    def mark_speech_event(self, is_assurance=False):
        """
        MUST be called whenever TTS is triggered.
        """
        with self.lock:
            now = time.time()
            self.last_spoken_timestamp = now
            if is_assurance:
                self.last_assurance_timestamp = now

    # ------------------------
    # ASSURANCE LOGIC
    # ------------------------
    def should_trigger_assurance(self, stable_people_count, timeout=30.0):
        with self.lock:
            now = time.time()

            if stable_people_count == 0:
                return False

            if now - self.last_spoken_timestamp < timeout:
                return False

            if now - self.last_assurance_timestamp < 25.0:
                return False

            return True

    # ------------------------
    # READ CONTEXT
    # ------------------------
    def get_context(self):
        with self.lock:
            now = time.time()

            caption = (
                self.last_caption
                if now - self.caption_timestamp < self.ttl
                else None
            )

            objects = (
                self.grounded_objects
                if now - self.objects_timestamp < self.ttl
                else []
            )

            return {
                "caption": caption,
                "objects": objects,
                "time_since_speech": now - self.last_spoken_timestamp,
            }
