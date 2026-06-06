import speech_recognition as sr
import threading
import queue
import subprocess

# ─────────────────────────────
# Shared State & Callbacks
# ─────────────────────────────
speech_callback = None  # Called when mic transcribes speech
tts_queue = queue.Queue()


def set_speech_callback(func):
    global speech_callback
    speech_callback = func


# ─────────────────────────────
# Text-to-Speech (Windows PowerShell)
# ─────────────────────────────
def speak_text(text):
    safe_text = text.replace('"', "'")
    subprocess.run(
        [
            "powershell", "-Command",
            f'Add-Type -AssemblyName System.Speech; '
            f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            f'$s.Speak("{safe_text}")'
        ],
        creationflags=subprocess.CREATE_NO_WINDOW
    )


def _tts_worker():
    """Background thread: drain the TTS queue sequentially."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        speak_text(text)
        tts_queue.task_done()


# ─────────────────────────────
# Microphone Speech Recognition
# ─────────────────────────────
def _mic_worker():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5        # wait longer before cutting off
    recognizer.phrase_threshold = 0.3       # min seconds of speech to count
    recognizer.non_speaking_duration = 0.8  # silence padding around speech
    recognizer.dynamic_energy_threshold = False  # stop auto-adjusting
    recognizer.energy_threshold = 400       # stable fixed threshold

    mic = sr.Microphone()

    print("[MIC] Calibrating microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)

    print("[MIC] Listening...")

    while True:
        try:
            with mic as source:
                print("🎤 Waiting...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)  # no hard cut-off

            print("[MIC] Transcribing...")
            text = recognizer.recognize_google(audio)
            print(f"✅ [MIC] {text}")

            # Speak the transcribed text aloud
            tts_queue.put(text)

            # Send to web UI
            if speech_callback:
                speech_callback(text)

        except sr.UnknownValueError:
            print("[MIC] Could not understand audio")
        except sr.RequestError as e:
            print(f"[MIC] Speech API error: {e}")


# ─────────────────────────────
# Start Everything
# ─────────────────────────────
def start_all(app_speech_callback):
    set_speech_callback(app_speech_callback)

    # TTS worker thread
    threading.Thread(target=_tts_worker, daemon=True).start()

    # Microphone thread
    threading.Thread(target=_mic_worker, daemon=True).start()