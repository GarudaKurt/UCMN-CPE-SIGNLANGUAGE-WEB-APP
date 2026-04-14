import speech_recognition as sr
import requests
import asyncio
import threading
import queue
import subprocess
from bleak import BleakClient, BleakScanner

# ─────────────────────────────
# ESP32 WiFi Config (Voice → ESP32)
# ─────────────────────────────
ESP32_IP = "192.168.1.100"
ENDPOINT = f"http://{ESP32_IP}:80/message"

# ─────────────────────────────
# BLE Config (ESP32 Flex Sensor → Flask)
# ─────────────────────────────
DEVICE_NAME = "ESP32-FlexSensor"
NOTIFY_UUID = "6E400003-B5A4-F393-E0A9-E50E24DCCA9E"

# ─────────────────────────────
# Shared State & Callbacks
# ─────────────────────────────
speech_callback = None        # Called when mic transcribes speech
ble_callback = None           # Called when BLE gesture text arrives
ble_status_callback = None    # Called when BLE connection status changes

speech_tts_queue = queue.Queue()
last_spoken_gesture = ""
ble_connected = False


def set_callback(func):
    """Register callback for microphone speech → web UI."""
    global speech_callback
    speech_callback = func


def set_ble_callback(func):
    """Register callback for BLE gesture text → web UI."""
    global ble_callback
    ble_callback = func


def set_ble_status_callback(func):
    """Register callback for BLE connection status → web UI."""
    global ble_status_callback
    ble_status_callback = func


# ─────────────────────────────
# ESP32 WiFi: Send speech text
# ─────────────────────────────
def send_to_esp32(text):
    try:
        response = requests.post(
            ENDPOINT,
            json={"text": text},
            timeout=5
        )
        print(f"[SENT] {text}")
        print(f"[ESP32] Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot reach ESP32 via WiFi")
    except requests.exceptions.Timeout:
        print("[ERROR] ESP32 WiFi Timeout")


# ─────────────────────────────
# Text-to-Speech (Windows PowerShell)
# ─────────────────────────────
def speak_text(text):
    """Use Windows built-in PowerShell TTS — no pyttsx3 issues."""
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
        text = speech_tts_queue.get()
        if text is None:
            break
        speak_text(text)
        speech_tts_queue.task_done()


# ─────────────────────────────
# BLE: Handle Notifications from ESP32 Flex Sensor
# ─────────────────────────────
def _handle_ble_notification(sender, data):
    global last_spoken_gesture
    message = data.decode("utf-8").strip()

    if message == "RESET":
        last_spoken_gesture = ""
        print("\n[BLE] Ready for next gesture")
        if ble_callback:
            ble_callback("__RESET__")
        return

    if message and message != last_spoken_gesture:
        print(f"\n[BLE Gesture] {message}")
        last_spoken_gesture = message

        # Speak the gesture label aloud
        speech_tts_queue.put(message)

        # Send to web UI
        if ble_callback:
            ble_callback(message)


async def _ble_listener():
    global ble_connected

    print(f"[BLE] Scanning for {DEVICE_NAME}...")
    if ble_status_callback:
        ble_status_callback("scanning")

    while True:
        try:
            device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)
            if not device:
                print("[BLE] Device not found, retrying...")
                if ble_status_callback:
                    ble_status_callback("not_found")
                await asyncio.sleep(5)
                continue

            async with BleakClient(device) as client:
                ble_connected = True
                print(f"[BLE] Connected to {DEVICE_NAME}")
                if ble_status_callback:
                    ble_status_callback("connected")

                await client.start_notify(NOTIFY_UUID, _handle_ble_notification)

                while client.is_connected:
                    await asyncio.sleep(1)

        except Exception as e:
            print(f"[BLE] Error: {e}")
            ble_connected = False
            if ble_status_callback:
                ble_status_callback("disconnected")
            await asyncio.sleep(5)


def start_ble_listener():
    """Start BLE listener in its own thread with its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_ble_listener())


# ─────────────────────────────
# Microphone Speech Recognition
# ─────────────────────────────
def start_listening():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    mic = sr.Microphone()

    print("[MIC] Calibrating microphone...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)

    print("[MIC] Listening...")

    while True:
        try:
            with mic as source:
                print("🎤 Waiting...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)

            print("[MIC] Transcribing...")
            text = recognizer.recognize_google(audio)
            print(f"✅ [MIC] {text}")

            # Send transcribed speech to ESP32 via WiFi
            send_to_esp32(text)

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
def start_all(app_speech_callback, app_ble_callback, app_ble_status_callback):
    set_callback(app_speech_callback)
    set_ble_callback(app_ble_callback)
    set_ble_status_callback(app_ble_status_callback)

    # TTS worker thread
    threading.Thread(target=_tts_worker, daemon=True).start()

    # BLE listener thread
    threading.Thread(target=start_ble_listener, daemon=True).start()

    # Microphone thread
    threading.Thread(target=start_listening, daemon=True).start()