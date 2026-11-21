import time
from typing import Tuple


class SemanticTrigger:
    """
    Decides when heavy semantic processing should occur.
    CONTRACT:
      should_trigger(...) MUST return exactly (bool, reason)
    """

    def __init__(self, min_trigger_interval=2.0):
        self.min_trigger_interval = min_trigger_interval
        self.last_trigger_time = 0.0
        self.last_stable_set = set()

    def should_trigger(self, accumulator_state) -> Tuple[bool, str]:
        now = time.time()

        # Rate limit safety
        if now - self.last_trigger_time < self.min_trigger_interval:
            return False, "rate_limited"

        # 1️⃣ New class appeared
        if accumulator_state.new_classes:
            self.last_trigger_time = now
            return True, f"new_class: {', '.join(accumulator_state.new_classes)}"

        current_stable = set(accumulator_state.stable_objects)

        # 2️⃣ Stable scene change
        if current_stable and current_stable != self.last_stable_set:
            self.last_trigger_time = now
            self.last_stable_set = current_stable
            return True, f"scene_stable: {', '.join(sorted(current_stable))}"

        return False, "no_trigger"
