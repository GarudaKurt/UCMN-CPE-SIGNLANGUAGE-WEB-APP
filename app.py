from flask import Flask, render_template, jsonify, request
import threading
import speech_worker

app = Flask(__name__)

latest_text = "Waiting for speech..."

# ─────────────────────────────
# Receive speech from worker
# ─────────────────────────────
def update_text(text):
    global latest_text
    latest_text = text

# Register callback
speech_worker.set_callback(update_text)

# Start speech recognition thread
threading.Thread(
    target=speech_worker.start_listening,
    daemon=True
).start()

# ─────────────────────────────
# Routes
# ─────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_text")
def get_text():
    return jsonify({"text": latest_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)