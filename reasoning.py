import ollama
import time


# -----------------------------
# Public constants
# -----------------------------

FALLBACK_SENTENCE = "People are present in front of you."


# -----------------------------
# Primary LLM-based speech
# -----------------------------

def generate_speech(
    speech_type,
    objects=None,
    caption=None,
    trigger_reason=None,
    people_count=0,
    timeout_seconds=5.0
):
    """
    Generate human-facing speech using LLM (slow but semantic).

    This is the PRIMARY pathway used by the pipeline.

    Args:
        speech_type: "event" or "assurance"
        objects: List[str]
        caption: str
        trigger_reason: str
        people_count: int
        timeout_seconds: float

    Returns:
        str | None
    """

    try:
        if speech_type == "event":
            prompt = _build_event_prompt(objects, caption, trigger_reason)
        elif speech_type == "assurance":
            prompt = _build_assurance_prompt(people_count)
        else:
            return None

        start = time.time()

        response = ollama.generate(
            model="llama3.2:1b",
            prompt=prompt,
            options={
                "temperature": 0.7,
                "num_predict": 50
            }
        )

        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            print(f"⚠️ LLM exceeded timeout ({elapsed:.2f}s)")
            return None

        result = response.get("response", "").strip()

        if not result or len(result) < 5:
            return None

        return result

    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        return None


# -----------------------------
# OPTIONAL fast (no-LLM) speech
# -----------------------------

def generate_fast_speech(
    speech_type,
    objects=None,
    people_count=0
):
    """
    Deterministic, zero-latency speech.
    NOT required by pipeline unless explicitly called.

    Safe to keep even if unused.
    """

    if speech_type == "assurance":
        if people_count == 1:
            return "One person is in front of you."
        elif people_count > 1:
            return f"{people_count} people are in front of you."
        else:
            return None

    if speech_type == "event" and objects:
        primary = objects[0]
        return f"{primary} detected nearby."

    return None


# -----------------------------
# Prompt builders
# -----------------------------

def _build_event_prompt(objects, caption, trigger_reason):
    objects_str = ", ".join(objects) if objects else "none"
    caption_str = caption if caption else "no description available"

    context = ""
    if trigger_reason:
        context = f"\nTrigger context: {trigger_reason}"

    return f"""
You are assisting a blind person in real-time.

Objects detected: {objects_str}
Scene description: {caption_str}{context}

Provide ONE calm, concise sentence describing what's happening right now. Do not mention 'Blind person' or 'Assistant'.
Maximum 12 words. Do not be descriptive. Do not be imaginative.
Do not speculate. Be precise and reassuring.
""".strip()


def _build_assurance_prompt(people_count):
    if people_count == 1:
        desc = "One person is in front of you"
    else:
        desc = f"{people_count} people are in front of you"

    return f"""
You are assisting a blind person in real-time.

{desc}

Provide ONE brief, reassuring sentence confirming presence.
Do not describe details.
""".strip()
