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
# Text-to-Speech (Windows PowerShell) — Filipino voice
# ─────────────────────────────
def speak_text(text):
    safe_text = text.replace('"', "'")

    # Try to use a Filipino/Tagalog voice first.
    # Falls back to default system voice if none is installed.
    ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

# Try to find a Filipino or Tagalog voice
$voices = $synth.GetInstalledVoices() | ForEach-Object {{ $_.VoiceInfo }}
$filVoice = $voices | Where-Object {{
    $_.Name -match "Filipino" -or
    $_.Name -match "Tagalog" -or
    $_.Culture -match "fil" -or
    $_.Culture -match "tl"
}} | Select-Object -First 1

if ($filVoice) {{
    $synth.SelectVoice($filVoice.Name)
    Write-Host "[TTS] Using voice: $($filVoice.Name)"
}} else {{
    Write-Host "[TTS] No Filipino voice found, using default voice."
}}

$synth.Speak("{safe_text}")
'''

    subprocess.run(
        ["powershell", "-Command", ps_script],
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
    recognizer.pause_threshold = 1.5
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.8
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 400

    mic = sr.Microphone()

    print("[MIC] Calibrating microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)

    print("[MIC] Listening...")

    while True:
        try:
            with mic as source:
                print("🎤 Waiting...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)

            print("[MIC] Transcribing...")
            text = recognizer.recognize_google(audio, language="fil-PH")
            print(f"✅ [MIC] {text}")

            # Speak the transcribed text aloud in Filipino voice
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