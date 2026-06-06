"""
fsl_worker.py
─────────────
Runs the FSL hand-landmark detector in a background thread.
Streams annotated JPEG frames via a generator (for Flask MJPEG route)
and fires a callback whenever a new sign is detected.
"""

import os
import time
import threading
import urllib.request
from collections import deque, Counter
import warnings

import cv2
import joblib
import mediapipe as mp
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ─────────────────────────────
# Paths & constants
# ─────────────────────────────
MODEL_PATH             = "hand_landmarker.task"
MODEL_URL              = ("https://storage.googleapis.com/mediapipe-models/"
                          "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
ALPHABET_MODEL_PATH    = "fsl_model.joblib"
MOTION_MODEL_PATH      = "fsl_motion_model.joblib"
NUMBER_MODEL_PATH      = "fsl_number_model.joblib"
PHRASE_MODEL_PATH      = "fsl_phrase_model.joblib"

CAMERA_INDEX                = 0
SEQUENCE_LENGTH             = 30
STATIC_CONFIDENCE_THRESHOLD = 0.60
MOTION_CONFIDENCE_THRESHOLD = 0.75
MOTION_MOVEMENT_THRESHOLD   = 0.15
NUMBER_CONFIDENCE_THRESHOLD = 0.60
PHRASE_CONFIDENCE_THRESHOLD = 0.70
PHRASE_MOVEMENT_THRESHOLD   = 0.10
MOTION_HOLD_SECONDS         = 2.0
MENU_HOLD_SECONDS           = 1.2

MODE_MENU     = "MENU"
MODE_ALPHABET = "ALPHABET"
MODE_NUMBERS  = "NUMBERS"
MODE_PHRASES  = "PHRASES"

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ─────────────────────────────
# Shared state
# ─────────────────────────────
_latest_frame_jpg: bytes = b""
_frame_lock = threading.Lock()

_detection_callback = None   # fired whenever a new sign is detected
_current_mode       = MODE_MENU
_mode_lock          = threading.Lock()


def set_detection_callback(fn):
    global _detection_callback
    _detection_callback = fn


def set_mode(mode: str):
    global _current_mode
    with _mode_lock:
        _current_mode = mode
    print(f"[FSL] Mode set to: {mode}")


def get_mode() -> str:
    with _mode_lock:
        return _current_mode


def get_latest_frame() -> bytes:
    with _frame_lock:
        return _latest_frame_jpg


def frame_generator():
    """MJPEG generator for Flask streaming route."""
    # Build a small black placeholder JPEG for when camera isn't ready yet
    placeholder = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()

    while True:
        frame = get_latest_frame()
        data  = frame if frame else placeholder
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.033)   # ~30 fps


def frame_sse_generator():
    """SSE generator that pushes base64-encoded JPEG frames.
    More reliable than MJPEG on Windows/Chrome."""
    import base64
    placeholder = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
    last_frame = None

    while True:
        frame = get_latest_frame()
        # Only push if frame changed
        if frame and frame != last_frame:
            last_frame = frame
            b64 = base64.b64encode(frame).decode("utf-8")
            yield f"data: {b64}\n\n"
        elif not frame:
            b64 = base64.b64encode(placeholder).decode("utf-8")
            yield f"data: {b64}\n\n"
        time.sleep(0.05)  # ~20 fps is plenty for a web UI


# ─────────────────────────────
# ML helpers (unchanged from original)
# ─────────────────────────────
def normalize_landmarks(hand_landmarks):
    wrist = hand_landmarks[0]
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    scale = max(max(xs)-min(xs), max(ys)-min(ys), 1e-6)
    features = []
    for lm in hand_landmarks:
        features.extend([
            (lm.x-wrist.x)/scale,
            (lm.y-wrist.y)/scale,
            (lm.z-wrist.z)/scale,
        ])
    return features


def raw_landmarks(hand_landmarks):
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]


def sequence_to_features(sequence):
    first_frame = sequence[0]
    wrist0 = first_frame[0]
    xs = [p[0] for p in first_frame]
    ys = [p[1] for p in first_frame]
    scale = max(max(xs)-min(xs), max(ys)-min(ys), 1e-6)
    features = []
    for frame in sequence:
        for x, y, z in frame:
            features.extend([
                (x-wrist0[0])/scale,
                (y-wrist0[1])/scale,
                (z-wrist0[2])/scale,
            ])
    return features


def calculate_movement(sequence):
    if len(sequence) < 2:
        return 0
    first_frame = sequence[0]
    xs = [p[0] for p in first_frame]
    ys = [p[1] for p in first_frame]
    scale = max(max(xs)-min(xs), max(ys)-min(ys), 1e-6)
    important = [0,4,8,12,16,20]
    max_dist = 0
    for frame in sequence:
        for pid in important:
            dx = (frame[pid][0]-first_frame[pid][0])/scale
            dy = (frame[pid][1]-first_frame[pid][1])/scale
            dist = (dx*dx+dy*dy)**0.5
            max_dist = max(max_dist, dist)
    return max_dist


def get_stable_prediction(history):
    if not history:
        return ""
    return Counter(history).most_common(1)[0][0]


def predict_static(classifier, hand_landmarks):
    features = np.array(normalize_landmarks(hand_landmarks)).reshape(1, -1)
    probs = classifier.predict_proba(features)[0]
    idx   = np.argmax(probs)
    return classifier.classes_[idx], probs[idx]


def predict_motion(classifier, sequence):
    features = np.array(sequence_to_features(sequence)).reshape(1, -1)
    probs = classifier.predict_proba(features)[0]
    idx   = np.argmax(probs)
    return classifier.classes_[idx], probs[idx]


def is_finger_up(lms, tip, pip):
    return lms[tip].y < lms[pip].y


def detect_menu_option(lms):
    iu = is_finger_up(lms, 8, 6)
    mu = is_finger_up(lms, 12, 10)
    ru = is_finger_up(lms, 16, 14)
    pu = is_finger_up(lms, 20, 18)
    if iu and not mu and not ru and not pu:
        return "1"
    if iu and mu and not ru and not pu:
        return "2"
    if iu and mu and ru and not pu:
        return "3"
    return ""


def draw_hand(frame, hand_landmarks, w, h):
    pts = []
    for lm in hand_landmarks:
        x, y = int(lm.x*w), int(lm.y*h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 5, (0,255,0), -1)
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (255,0,0), 2)


def overlay_text(frame, lines, y_start=30, color=(0,255,0)):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20, y_start + i*35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# ─────────────────────────────
# Main detection loop
# ─────────────────────────────
def _detection_loop():
    global _latest_frame_jpg, _current_mode

    # ── Download MediaPipe model if missing ──
    if not os.path.exists(MODEL_PATH):
        print("[FSL] Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    # ── Load ML models ──
    clf_alphabet = joblib.load(ALPHABET_MODEL_PATH) if os.path.exists(ALPHABET_MODEL_PATH) else None
    clf_motion   = joblib.load(MOTION_MODEL_PATH)   if os.path.exists(MOTION_MODEL_PATH)   else None
    clf_number   = joblib.load(NUMBER_MODEL_PATH)   if os.path.exists(NUMBER_MODEL_PATH)   else None
    clf_phrase   = joblib.load(PHRASE_MODEL_PATH)   if os.path.exists(PHRASE_MODEL_PATH)   else None

    # ── MediaPipe setup ──
    BaseOptions       = mp.tasks.BaseOptions
    HandLandmarker    = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )

    # ── State ──
    prediction_history = deque(maxlen=10)
    motion_sequence    = deque(maxlen=SEQUENCE_LENGTH)
    phrase_sequence    = deque(maxlen=SEQUENCE_LENGTH)
    last_detected      = ""
    motion_start_time  = None
    menu_hold_start    = None
    menu_hold_option   = ""

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[FSL] ERROR: Cannot open camera.")
        return

    print("[FSL] Camera opened. Starting detection loop...")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            mode  = get_mode()
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            hand_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None

            # ── Draw skeleton ──
            if hand_landmarks:
                draw_hand(frame, hand_landmarks, w, h)

            # ── Per-mode logic ──
            if mode == MODE_MENU:
                if hand_landmarks:
                    opt = detect_menu_option(hand_landmarks)
                    if opt:
                        if opt == menu_hold_option:
                            if menu_hold_start and (time.time() - menu_hold_start) >= MENU_HOLD_SECONDS:
                                new_mode = {
                                    "1": MODE_ALPHABET,
                                    "2": MODE_NUMBERS,
                                    "3": MODE_PHRASES,
                                }.get(opt, MODE_MENU)
                                set_mode(new_mode)
                                if _detection_callback:
                                    _detection_callback({"label": new_mode, "mode": "MENU"})
                                menu_hold_start  = None
                                menu_hold_option = ""
                        else:
                            menu_hold_option = opt
                            menu_hold_start  = time.time()
                    else:
                        menu_hold_option = ""
                        menu_hold_start  = None

                overlay_text(frame, [
                    "MENU MODE",
                    "1 finger = ALPHABET",
                    "2 fingers = NUMBERS",
                    "3 fingers = PHRASES",
                ])

            elif mode == MODE_ALPHABET and clf_alphabet:
                if hand_landmarks:
                    label, conf = predict_static(clf_alphabet, hand_landmarks)
                    if conf >= STATIC_CONFIDENCE_THRESHOLD:
                        prediction_history.append(label)
                stable = get_stable_prediction(prediction_history)
                if stable and stable != last_detected:
                    last_detected = stable
                    if _detection_callback:
                        _detection_callback({"label": stable, "mode": "ALPHABET"})
                overlay_text(frame, [f"ALPHABET: {stable}", f"conf: {conf:.0%}" if hand_landmarks else ""])

            elif mode == MODE_NUMBERS and clf_number:
                if hand_landmarks:
                    label, conf = predict_static(clf_number, hand_landmarks)
                    if conf >= NUMBER_CONFIDENCE_THRESHOLD:
                        prediction_history.append(label)
                stable = get_stable_prediction(prediction_history)
                if stable and stable != last_detected:
                    last_detected = stable
                    if _detection_callback:
                        _detection_callback({"label": stable, "mode": "NUMBERS"})
                overlay_text(frame, [f"NUMBERS: {stable}"])

            elif mode == MODE_PHRASES and clf_phrase:
                if hand_landmarks:
                    lms = raw_landmarks(hand_landmarks)
                    phrase_sequence.append(lms)

                if len(phrase_sequence) == SEQUENCE_LENGTH:
                    movement = calculate_movement(list(phrase_sequence))
                    if movement >= PHRASE_MOVEMENT_THRESHOLD:
                        label, conf = predict_motion(clf_phrase, list(phrase_sequence))
                        if conf >= PHRASE_CONFIDENCE_THRESHOLD and label != last_detected:
                            last_detected = label
                            if _detection_callback:
                                _detection_callback({"label": label, "mode": "PHRASES"})
                overlay_text(frame, ["PHRASES MODE"])

            # ── Mode label on frame ──
            cv2.putText(frame, f"MODE: {mode}", (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # ── Encode and publish frame ──
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with _frame_lock:
                _latest_frame_jpg = jpg.tobytes()

    cap.release()

def start(detection_callback=None):
    """Call once from app.py to start the FSL background thread."""
    if detection_callback:
        set_detection_callback(detection_callback)
    t = threading.Thread(target=_detection_loop, daemon=True)
    t.start()
    print("[FSL] Detection thread started.")