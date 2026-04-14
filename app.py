from flask import Flask, render_template, jsonify, request, Response
import threading
import queue
import json
import speech_worker

app = Flask(__name__)

# ─────────────────────────────
# Shared State
# ─────────────────────────────
latest_speech_text = "Waiting for speech..."
latest_gesture_text = "Waiting for gesture..."
ble_status = "disconnected"

# SSE client queues — one per connected browser
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()


def _broadcast(event_type: str, data: dict):
    """Push a Server-Sent Event to every connected browser tab."""
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


# ─────────────────────────────
# Callbacks from speech_worker
# ─────────────────────────────
def on_speech(text: str):
    global latest_speech_text
    latest_speech_text = text
    _broadcast("speech", {"text": text})


def on_gesture(text: str):
    global latest_gesture_text
    if text == "__RESET__":
        latest_gesture_text = "Ready for next gesture..."
        _broadcast("gesture", {"text": "", "reset": True})
    else:
        latest_gesture_text = text
        _broadcast("gesture", {"text": text})


def on_ble_status(status: str):
    global ble_status
    ble_status = status
    _broadcast("ble_status", {"status": status})


# ─────────────────────────────
# Start all background workers
# ─────────────────────────────
speech_worker.start_all(on_speech, on_gesture, on_ble_status)

# ─────────────────────────────
# Routes
# ─────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_state")
def get_state():
    """Initial state snapshot for page load."""
    return jsonify({
        "speech": latest_speech_text,
        "gesture": latest_gesture_text,
        "ble_status": ble_status,
    })


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — pushes updates to browser in real-time."""
    client_q: queue.Queue = queue.Queue(maxsize=50)

    with _sse_lock:
        _sse_clients.append(client_q)

    def generate():
        # Send current state immediately on connect
        yield f"event: init\ndata: {json.dumps({'speech': latest_speech_text, 'gesture': latest_gesture_text, 'ble_status': ble_status})}\n\n"
        try:
            while True:
                try:
                    msg = client_q.get(timeout=20)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"   # prevent proxy timeouts
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                if client_q in _sse_clients:
                    _sse_clients.remove(client_q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/speak", methods=["POST"])
def speak_manual():
    """Manually trigger TTS from the web UI."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if text:
        import subprocess
        safe_text = text.replace('"', "'")
        threading.Thread(
            target=lambda: subprocess.run(
                ["powershell", "-Command",
                 f'Add-Type -AssemblyName System.Speech; '
                 f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                 f'$s.Speak("{safe_text}")'],
                creationflags=subprocess.CREATE_NO_WINDOW
            ),
            daemon=True
        ).start()
        return jsonify({"ok": True, "text": text})
    return jsonify({"ok": False, "error": "No text provided"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)