from flask import Flask, render_template, jsonify, request, Response
import threading
import queue
import json
import base64
import time
import subprocess
import sys
import os
import speech_worker
import fsl_worker

app = Flask(__name__)

# ─────────────────────────────
# Shared State
# ─────────────────────────────
latest_speech_text = "Waiting for speech..."
latest_fsl_label   = ""
latest_fsl_mode    = ""

# SSE client queues
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
# Routes — core
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
        "frame_size":   len(frame) if frame else 0,
    })


# ─────────────────────────────
# Routes — phrase model
# ─────────────────────────────

@app.route("/phrase_model_info")
def phrase_model_info():
    """Return info about the currently loaded phrase model."""
    return jsonify(fsl_worker.get_phrase_model_info())


@app.route("/reload_phrase_model", methods=["POST"])
def reload_phrase_model():
    """
    Hot-swap the phrase model without restarting Flask.
    Call this after train_phrases.py finishes.
    """
    result = fsl_worker.reload_phrase_model()
    status = 200 if result["ok"] else 500
    return jsonify(result), status


@app.route("/train_phrases", methods=["POST"])
def train_phrases():
    """
    Kick off train_phrases.py in a subprocess and stream its output as SSE.
    The browser can listen on /train_phrases_stream to watch progress,
    or just POST here and poll /phrase_model_info.

    Returns immediately with {"ok": true, "started": true}.
    Training output is broadcast as SSE "train_log" events.
    """
    def _run_training():
        try:
            proc = subprocess.Popen(
                [sys.executable, "train_phrases.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip()
                print(f"[TRAIN] {line}")
                _broadcast("train_log", {"line": line})
            proc.wait()
            if proc.returncode == 0:
                # Auto hot-reload after successful training
                reload_result = fsl_worker.reload_phrase_model()
                _broadcast("train_done", {
                    "success":  True,
                    "classes":  reload_result.get("classes", []),
                })
            else:
                _broadcast("train_done", {
                    "success": False,
                    "error":   "Training process exited with non-zero code",
                })
        except Exception as exc:
            _broadcast("train_done", {"success": False, "error": str(exc)})

    t = threading.Thread(target=_run_training, daemon=True)
    t.start()
    return jsonify({"ok": True, "started": True})


@app.route("/phrase_data_stats")
def phrase_data_stats():
    """
    Return how many samples have been collected per phrase label.
    Reads phrase_data/ directory structure.
    """
    data_dir = "phrase_data"
    stats = {}
    if os.path.isdir(data_dir):
        for folder in sorted(os.listdir(data_dir)):
            fp = os.path.join(data_dir, folder)
            if os.path.isdir(fp):
                label   = folder.replace("_", " ")
                n_files = len([f for f in os.listdir(fp) if f.endswith(".npy")])
                stats[label] = n_files
    return jsonify({"ok": True, "stats": stats})


# ─────────────────────────────
# SSE endpoint
# ─────────────────────────────

@app.route("/stream")
def stream():
    """
    Single SSE endpoint pushing:
      - speech events
      - fsl detection events
      - camera frame events (base64 JPEG) at ~20 fps
      - train_log / train_done events
    """
    client_q: queue.Queue = queue.Queue(maxsize=200)
    with _sse_lock:
        _sse_clients.append(client_q)

    def generate():
        yield (
            f"event: init\ndata: {json.dumps({'speech': latest_speech_text, 'fsl_label': latest_fsl_label, 'fsl_mode': latest_fsl_mode})}\n\n"
        )

        last_frame_bytes = None
        last_frame_time  = 0
        FRAME_INTERVAL   = 0.05   # 20 fps max

        try:
            while True:
                now = time.time()

                # Drain pending events (up to 5 at a time)
                drained = 0
                while drained < 5:
                    try:
                        msg = client_q.get_nowait()
                        yield msg
                        drained += 1
                    except queue.Empty:
                        break

                # Push new camera frame if interval elapsed and frame changed
                if now - last_frame_time >= FRAME_INTERVAL:
                    frame = fsl_worker.get_latest_frame()
                    if frame and frame != last_frame_bytes:
                        last_frame_bytes = frame
                        last_frame_time  = now
                        b64 = base64.b64encode(frame).decode("utf-8")
                        yield f"event: frame\ndata: {b64}\n\n"
                    elif not frame:
                        last_frame_time = now
                        yield ": keepalive\n\n"

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