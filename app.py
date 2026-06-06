from flask import Flask, render_template, jsonify, request, Response
import threading
import queue
import json
import base64
import time
import speech_worker
import fsl_worker

app = Flask(__name__)

# ─────────────────────────────
# Shared State
# ─────────────────────────────
latest_speech_text = "Waiting for speech..."
latest_fsl_label   = ""
latest_fsl_mode    = ""

# SSE client queues — one per connected browser tab
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()


def _broadcast(event_type: str, data: dict):
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
# Callbacks
# ─────────────────────────────
def on_speech(text: str):
    global latest_speech_text
    latest_speech_text = text
    _broadcast("speech", {"text": text})


def on_fsl_detection(result: dict):
    global latest_fsl_label, latest_fsl_mode
    latest_fsl_label = result["label"]
    latest_fsl_mode  = result["mode"]
    _broadcast("fsl", result)


# ─────────────────────────────
# Start background workers
# ─────────────────────────────
speech_worker.start_all(on_speech)
fsl_worker.start(on_fsl_detection)

# ─────────────────────────────
# Routes
# ─────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_state")
def get_state():
    return jsonify({
        "speech":           latest_speech_text,
        "fsl_label":        latest_fsl_label,
        "fsl_mode":         latest_fsl_mode,
        "fsl_mode_current": fsl_worker.get_mode(),
    })

@app.route("/video_feed")
def video_feed():
    return Response(
        fsl_worker.frame_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/set_fsl_mode", methods=["POST"])
def set_fsl_mode():
    data = request.get_json(force=True)
    mode = data.get("mode", "").upper()
    valid = {"MENU", "ALPHABET", "NUMBERS", "PHRASES"}
    if mode in valid:
        fsl_worker.set_mode(mode)
        return jsonify({"ok": True, "mode": mode})
    return jsonify({"ok": False, "error": "Invalid mode"}), 400

@app.route("/debug_frame")
def debug_frame():
    frame = fsl_worker.get_latest_frame()

    return jsonify({
        "frame_exists": frame is not None,
        "frame_size": len(frame) if frame else 0
    })
@app.route("/stream")
def stream():
    """
    Single SSE endpoint that pushes:
      - speech events
      - fsl detection events
      - camera frame events (base64 JPEG) at ~20 fps
    Replaces the old MJPEG /video_feed route entirely.
    """
    client_q: queue.Queue = queue.Queue(maxsize=100)
    with _sse_lock:
        _sse_clients.append(client_q)

    def generate():
        # Send initial state immediately
        yield (
            f"event: init\ndata: {json.dumps({'speech': latest_speech_text, 'fsl_label': latest_fsl_label, 'fsl_mode': latest_fsl_mode})}\n\n"
        )

        last_frame_bytes = None
        last_frame_time  = 0
        FRAME_INTERVAL   = 0.05   # 20 fps max

        try:
            while True:
                now = time.time()

                # Drain any pending speech / fsl events (up to 5 at a time)
                drained = 0
                while drained < 5:
                    try:
                        msg = client_q.get_nowait()
                        yield msg
                        drained += 1
                    except queue.Empty:
                        break

                # Push a new camera frame if interval has elapsed and frame changed
                if now - last_frame_time >= FRAME_INTERVAL:
                    frame = fsl_worker.get_latest_frame()
                    if frame and frame != last_frame_bytes:
                        last_frame_bytes = frame
                        last_frame_time  = now
                        b64 = base64.b64encode(frame).decode("utf-8")
                        yield f"event: frame\ndata: {b64}\n\n"
                    elif not frame:
                        # No frame yet — send keepalive so connection stays open
                        last_frame_time = now
                        yield ": keepalive\n\n"

                # Short sleep to avoid spinning the CPU
                time.sleep(0.01)

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
            "Cache-Control":     "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)