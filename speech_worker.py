import speech_recognition as sr
import requests

ESP32_IP = "192.168.1.100"
ENDPOINT = f"http://{ESP32_IP}:80/message"

callback_function = None


def set_callback(func):
    global callback_function
    callback_function = func


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
        print("[ERROR] Cannot reach ESP32")

    except requests.exceptions.Timeout:
        print("[ERROR] ESP32 Timeout")


def start_listening():
    recognizer = sr.Recognizer()

    recognizer.pause_threshold = 0.8
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    mic = sr.Microphone()

    print("Calibrating microphone...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(
            source,
            duration=2
        )

    print("Listening...")

    while True:
        try:
            with mic as source:
                print("🎤 Waiting...")
                audio = recognizer.listen(
                    source,
                    timeout=None,
                    phrase_time_limit=15
                )

            print("Transcribing...")

            text = recognizer.recognize_google(audio)

            print(f"✅ {text}")

            # Send to ESP32
            send_to_esp32(text)

            # Send to Web UI
            if callback_function:
                callback_function(text)

        except sr.UnknownValueError:
            print("Could not understand")

        except sr.RequestError as e:
            print(f"Speech error: {e}")