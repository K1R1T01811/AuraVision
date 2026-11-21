import time

class TemporalSceneMemory:
    def __init__(self, ttl=8.0):
        """
        ttl: time (seconds) before same scene is allowed again
        """
        self.last_scene = None
        self.last_time = 0
        self.ttl = ttl

    def is_duplicate(self, scene):
        """
        Returns True if scene is same as last scene within TTL
        """
        now = time.time()

        if scene == self.last_scene and (now - self.last_time) < self.ttl:
            return True

        return False

    def update(self, scene):
        """
        Store new scene and timestamp
        """
        self.last_scene = scene
        self.last_time = time.time()

    def get_context(self):
        """
        Return recent scene context for temporal continuity
        """
        if self.last_scene is None:
            return {}
        
        time_since = time.time() - self.last_time
        return {
            "last_scene": self.last_scene,
            "time_since": time_since,
            "is_recent": time_since < self.ttl
        }

