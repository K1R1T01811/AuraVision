import subprocess
import threading
import time

# Public flag (USED BY vision worker)
tts_busy = False

# Internal lock
_lock = threading.Lock()


def _speak_worker(text: str):
    global tts_busy

    with _lock:
        tts_busy = True

    try:
        # Windows native TTS using PowerShell
        command = [
            "powershell",
            "-Command",
            f'''
            Add-Type -AssemblyName System.Speech;
            $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
            $speak.Rate = -1;
            $speak.Speak("{text}");
            '''
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False
        )

    except Exception as e:
        print("🔇 TTS Error:", e)

    time.sleep(1.2)  # natural pause between speeches

    with _lock:
        tts_busy = False


def speak(text: str):
    global tts_busy

    if not text:
        return

    # Prevent overlapping speech
    if tts_busy:
        return

    thread = threading.Thread(
        target=_speak_worker,
        args=(text,),
        daemon=True
    )
    thread.start()
