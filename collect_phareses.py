"""
collect_phrases.py
──────────────────
Interactive data collector for 25 common FSL Phrases.
Captures motion SEQUENCES (30 frames) per phrase — not static poses.

TWO-HAND SUPPORT
────────────────
num_hands=2 is now enabled.  Each frame stores landmarks for BOTH hands.
If only one hand is visible, the missing hand is zero-padded so every row
in the CSV always has the same number of features:
    30 frames × 2 hands × 21 landmarks × 3 coords = 3 780 features

Signs that use one hand (most phrases) will simply have zeros in the
second-hand slot — the model learns this as "no second hand present".
Signs that use two hands ("How are you", etc.) get real data in both slots.

DROPOUT TOLERANCE (NEW)
────────────────────────
The hand landmarker can briefly lose tracking for 1-2 frames even when
your hand is clearly in frame (motion blur, awkward angle, etc.). This is
especially common for fast or sharply-shaped signs like "I love you".

Previously: any frame with 0 hands detected was simply skipped, which
could stall a recording indefinitely if it kept happening, and a single
drop-out felt like "no hands detected" even though the sign was performed
correctly.

Now: if hands briefly disappear mid-sequence, the collector re-uses the
last known landmark frame for up to MAX_DROPOUT_FRAMES consecutive frames
so the sequence still completes. If the dropout lasts longer than that
(meaning the hand truly left the frame), the buffer is cleared and
recording waits for the hand to reappear.

Also: detection confidence thresholds are lowered slightly so fast,
sharply-angled signs (like "I love you", which has a distinct splayed
thumb/index/pinky shape) are tracked more reliably.

Usage:
    python collect_phrases.py

Controls (while the camera window is open):
    LEFT / RIGHT arrow  → previous / next phrase
    0-9                 → jump to phrase by index (0=phrase 1 … 9=phrase 10)
    SPACE               → start / pause recording for current phrase
    X / Backspace       → delete last saved sequence for current phrase
    S                   → print sample counts to terminal
    Q / ESC             → quit and save everything to data/phrases_dataset.csv

How recording works:
    When you press SPACE to start, the collector waits for at least one hand
    to appear, then captures exactly 30 frames of landmarks per sequence.
    Each sequence = one row in the CSV (label + 3780 features).
    Brief tracking dropouts (<= MAX_DROPOUT_FRAMES) no longer reset the
    sequence — the last good frame is reused instead.

Tips for good data:
    - Perform each phrase naturally and at normal speed
    - Vary your distance from camera and slight angle each rep
    - Aim for 80-120 sequences per phrase (the bar shows your progress)
    - For two-handed signs, make sure BOTH hands are clearly in frame
    - For "I love you", keep your hand facing the camera with fingers
      spread clearly — extreme side angles make the thumb/pinky harder
      to detect
"""

import os
import csv
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────
# Config
# ─────────────────────────────
MODEL_PATH   = "hand_landmarker.task"
MODEL_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
OUTPUT_DIR   = "data"
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "phrases_dataset.csv")
CAMERA_INDEX = 0

SEQUENCE_LENGTH = 30    # frames per sequence (must match fsl_worker.py)
SAMPLES_GOAL    = 100   # target sequences per phrase

# Detection confidence (lowered slightly to help with fast/angled signs
# such as "I love you", which can momentarily look ambiguous to the model)
MIN_HAND_DETECTION_CONFIDENCE = 0.4
MIN_HAND_PRESENCE_CONFIDENCE  = 0.4
MIN_TRACKING_CONFIDENCE       = 0.4

# How many consecutive "no hand detected" frames to tolerate during a
# recording before we give up and clear the buffer. Reusing the last good
# frame for short dropouts keeps fast signs like "I love you" from
# constantly restarting.
MAX_DROPOUT_FRAMES = 5

# ── Two-hand feature layout ────────────────────────────────────────────────
# Each frame: hand_0 (21×3=63 values) + hand_1 (21×3=63 values) = 126 values
# Full sequence: 30 × 126 = 3780 features
NUM_HANDS    = 2
PER_HAND     = 21 * 3          # 63 floats per hand
PER_FRAME    = NUM_HANDS * PER_HAND   # 126 floats per frame
NUM_FEATURES = SEQUENCE_LENGTH * PER_FRAME   # 3780

# ─────────────────────────────
# 25 Common FSL Phrases
# ─────────────────────────────
PHRASES = [
    "Hello",
    "Thank you",
    "Please",
    "Sorry",
    "Yes",
    "No",
    "Help me",
    "I love you",
    "Good morning",
    "Good night",
    "How are you",
    "I am fine",
    "What is your name",
    "My name is",
    "Nice to meet you",
    "Goodbye",
    "Come here",
    "Wait",
    "I don't understand",
    "Can you repeat",
    "Eat",
    "Drink",
    "Bathroom",
    "I need help",
    "You are welcome",
]

# Phrases known to use two hands — shown with a ✌ indicator in the HUD
TWO_HAND_PHRASES = {"How are you", "I love you", "Nice to meet you", "You are welcome"}

# ─────────────────────────────
# Colours (BGR)
# ─────────────────────────────
C_GREEN  = (0,  210,  80)
C_RED    = (30,  40, 220)
C_YELLOW = (0,  200, 200)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,     0,   0)
C_BLUE   = (210,  80,   0)
C_ORANGE = (0,  140, 255)
C_PURPLE = (200,  80, 200)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# Different colours for left vs right hand skeleton
HAND_COLORS = [C_BLUE, C_PURPLE]


# ─────────────────────────────
# Landmark helpers
# ─────────────────────────────
def raw_landmarks(hand_landmarks):
    """Return list of (x, y, z) tuples for one hand."""
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]


def frame_to_flat(hand_list):
    """
    Convert a list of up to 2 detected hands into a flat list of
    NUM_HANDS * 21 * 3 = 126 floats for one frame.

    Missing hands are zero-padded so the output length is always 126.
    hand_list: list of raw_landmarks() results (0, 1, or 2 items)
    """
    flat = []
    for i in range(NUM_HANDS):
        if i < len(hand_list):
            for x, y, z in hand_list[i]:
                flat.extend([x, y, z])
        else:
            flat.extend([0.0] * PER_HAND)   # pad absent hand with zeros
    return flat


def draw_hand(frame, hand_landmarks, w, h, color):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, C_GREEN, -1)
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], color, 2)


def put_text(frame, text, pos, scale=0.75, color=C_WHITE, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, C_BLACK, thickness + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thickness)


# ─────────────────────────────
# HUD
# ─────────────────────────────
def draw_hud(frame, phrase_idx, capturing, counts, hands_detected,
             frames_captured, last_saved_time, recording_ready,
             dropout_count):
    h, w = frame.shape[:2]
    phrase    = PHRASES[phrase_idx]
    count     = counts.get(phrase, 0)
    two_hand  = phrase in TWO_HAND_PHRASES
    hand_icon = " ✌" if two_hand else ""

    # ── top bar ──────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 15), -1)

    if capturing:
        if recording_ready:
            if dropout_count > 0:
                status_color = C_ORANGE
                status_text  = f"REC  [{frames_captured}/{SEQUENCE_LENGTH}]  (hold steady, {dropout_count}/{MAX_DROPOUT_FRAMES} drop)"
            else:
                status_color = C_GREEN
                status_text  = f"REC  [{frames_captured}/{SEQUENCE_LENGTH}]"
        else:
            status_color = C_ORANGE
            status_text  = "Waiting for hand..."
    else:
        status_color = C_YELLOW
        status_text  = "PAUSED"

    put_text(frame, status_text, (12, 40), 0.85, status_color)

    # phrase name + two-hand indicator
    put_text(frame, f"{phrase_idx+1}/25: {phrase}{hand_icon}", (220, 25), 0.65, C_WHITE)

    # progress bar
    bar_x = w - 260
    cv2.rectangle(frame, (bar_x, 15), (bar_x + 200, 45), (60, 60, 60), -1)
    fill = int(min(count / SAMPLES_GOAL, 1.0) * 200)
    bar_color = C_GREEN if count >= SAMPLES_GOAL else C_YELLOW
    if fill > 0:
        cv2.rectangle(frame, (bar_x, 15), (bar_x + fill, 45), bar_color, -1)
    put_text(frame, f"{count}/{SAMPLES_GOAL}", (w - 55, 38), 0.6, C_WHITE, 1)

    # ── sequence progress bar (while recording) ───────────────
    if capturing and recording_ready and frames_captured > 0:
        seq_w = int((frames_captured / SEQUENCE_LENGTH) * (w - 40))
        cv2.rectangle(frame, (20, 65), (20 + seq_w, 75), C_GREEN, -1)
        cv2.rectangle(frame, (20, 65), (w - 20, 75), C_WHITE, 1)

    # ── flash on save ─────────────────────────────────────────
    if last_saved_time and (time.time() - last_saved_time) < 0.4:
        cv2.rectangle(frame, (0, 0), (w, h), C_GREEN, 8)
        put_text(frame, "SEQUENCE SAVED", (w // 2 - 120, h // 2), 1.2, C_GREEN, 3)

    # ── hand status ───────────────────────────────────────────
    n = hands_detected
    if two_hand:
        hc = C_GREEN if n >= 2 else (C_ORANGE if n == 1 else C_RED)
        ht = f"Hands: {n}/2 detected" + ("  ← need BOTH hands" if n < 2 else "")
    else:
        hc = C_GREEN if n >= 1 else C_RED
        ht = f"Hand: {'detected' if n >= 1 else 'not found'}"
    put_text(frame, ht, (12, h - 18), 0.55, hc, 1)

    # ── two-hand reminder banner ──────────────────────────────
    if two_hand:
        cv2.rectangle(frame, (0, h - 115), (w, h - 95), (30, 30, 80), -1)
        put_text(frame, "TWO-HAND SIGN — keep both hands in frame", (12, h - 100), 0.48, C_PURPLE, 1)

    # ── controls ──────────────────────────────────────────────
    controls = [
        "← → : prev/next phrase",
        "SPACE: rec on/off",
        "X/Bksp: delete last",
        "S: show counts",
        "Q/ESC: quit & save",
    ]
    for i, txt in enumerate(controls):
        put_text(frame, txt, (12, h - 130 - i * 20), 0.45, C_WHITE, 1)

    # ── mini phrase list (right side) ─────────────────────────
    list_x = w - 210
    put_text(frame, "Phrases:", (list_x, 100), 0.5, C_YELLOW, 1)
    visible = range(max(0, phrase_idx - 3), min(len(PHRASES), phrase_idx + 8))
    for i, pidx in enumerate(visible):
        py = 122 + i * 20
        p  = PHRASES[pidx]
        c  = counts.get(p, 0)
        color  = C_GREEN if c >= SAMPLES_GOAL else (C_YELLOW if pidx == phrase_idx else C_WHITE)
        marker = ">" if pidx == phrase_idx else " "
        icon   = "✌" if p in TWO_HAND_PHRASES else " "
        label  = f"{marker}{pidx+1:2}.{icon}{p[:14]:<14} {c:>3}"
        put_text(frame, label, (list_x, py), 0.40, color, 1)


# ─────────────────────────────
# CSV helpers
# ─────────────────────────────
HEADER = ["label"] + [f"f{i}" for i in range(NUM_FEATURES)]


def load_existing(path):
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            # Check if this CSV was recorded with the OLD 1-hand format (1890 features)
            old_feature_count = SEQUENCE_LENGTH * 21 * 3   # 1890
            if len(existing_header) - 1 == old_feature_count:
                print(
                    f"\n[WARNING] Existing dataset was recorded with 1-hand format "
                    f"({old_feature_count} features).\n"
                    f"          New format uses 2-hand ({NUM_FEATURES} features).\n"
                    f"          Old data will be SKIPPED — please re-collect all phrases.\n"
                )
                return []   # cannot mix old and new format
            for row in reader:
                rows.append(row)
    return rows


def save_all(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def count_per_phrase(rows):
    counts = {p: 0 for p in PHRASES}
    for row in rows:
        lbl = row.get("label", "")
        if lbl in counts:
            counts[lbl] += 1
    return counts


def sequence_to_row(label, sequence):
    """
    Flatten a 30-frame sequence into a CSV row.
    Each element of `sequence` is the output of frame_to_flat():
    a list of 126 floats (2 hands × 21 landmarks × 3 coords).
    """
    flat = []
    for frame_flat in sequence:
        flat.extend(frame_flat)
    # Safety pad
    while len(flat) < NUM_FEATURES:
        flat.append(0.0)
    flat = flat[:NUM_FEATURES]

    row = {"label": label}
    row.update({f"f{i}": round(flat[i], 6) for i in range(NUM_FEATURES)})
    return row


# ─────────────────────────────
# Main
# ─────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Done.")

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,           # ← TWO-HAND DETECTION ENABLED
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    rows   = load_existing(OUTPUT_FILE)
    counts = count_per_phrase(rows)
    print(f"[INFO] Loaded {len(rows)} existing sequences.")
    for p in PHRASES:
        icon = "✌" if p in TWO_HAND_PHRASES else " "
        print(f"  {icon} {p}: {counts[p]} sequences")

    phrase_idx      = 0
    capturing       = False
    frame_buffer    = deque(maxlen=SEQUENCE_LENGTH)
    last_saved_time = None

    # Dropout-tolerance state
    last_hand_list  = None   # last known list of raw_landmarks() per hand
    dropout_count   = 0       # consecutive frames with 0 hands during recording

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("\n[INFO] Camera open.  TWO-HAND mode active.")
    print("  Use LEFT/RIGHT arrows to navigate phrases.")
    print("  Press SPACE to start/stop recording a phrase sequence.")
    print("  Press Q to quit and save.\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            # Collect all detected hands (up to 2)
            all_hands     = result.hand_landmarks   # list of 0–2 hand landmark lists
            hands_detected = len(all_hands)

            # recording_ready means: we are capturing AND either hands are
            # currently visible, OR we're still within the dropout-tolerance
            # window (so the sequence can continue using the last good frame).
            currently_has_hand = hands_detected >= 1
            within_dropout_window = (
                capturing
                and not currently_has_hand
                and last_hand_list is not None
                and dropout_count < MAX_DROPOUT_FRAMES
                and len(frame_buffer) > 0
            )
            recording_ready = capturing and (currently_has_hand or within_dropout_window)

            # ── Capture frame into buffer ─────────────────────
            if capturing:
                if currently_has_hand:
                    # Good frame — reset dropout counter, remember it
                    hand_list = [raw_landmarks(h_lm) for h_lm in all_hands]
                    last_hand_list = hand_list
                    dropout_count  = 0
                elif within_dropout_window:
                    # Brief dropout — reuse the last known landmarks
                    hand_list = last_hand_list
                    dropout_count += 1
                else:
                    hand_list = None

                if hand_list is not None:
                    flat_frame = frame_to_flat(hand_list)
                    frame_buffer.append(flat_frame)

                    # Once we have a full sequence, save it
                    if len(frame_buffer) == SEQUENCE_LENGTH:
                        phrase = PHRASES[phrase_idx]
                        row    = sequence_to_row(phrase, list(frame_buffer))
                        rows.append(row)
                        counts[phrase] = counts.get(phrase, 0) + 1
                        last_saved_time = time.time()
                        frame_buffer.clear()
                        last_hand_list = None
                        dropout_count  = 0
                        print(f"  [SAVED] '{phrase}'  total: {counts[phrase]}")
                elif capturing and not currently_has_hand:
                    # Hand truly gone too long — abandon this sequence and
                    # wait for a hand to reappear before starting a new one.
                    if len(frame_buffer) > 0:
                        print(f"  [DROPPED] Lost hand for too long, restarting sequence for '{PHRASES[phrase_idx]}'")
                    frame_buffer.clear()
                    last_hand_list = None
                    dropout_count  = 0
            else:
                # Not capturing — keep state clean
                last_hand_list = None
                dropout_count  = 0

            # ── Draw all detected hands ───────────────────────
            for i, hand_lm in enumerate(all_hands):
                draw_hand(frame, hand_lm, w, h, HAND_COLORS[i % 2])

            draw_hud(
                frame, phrase_idx, capturing, counts,
                hands_detected, len(frame_buffer),
                last_saved_time, recording_ready,
                dropout_count
            )

            cv2.imshow("FSL Phrase Collector  |  Q = quit", frame)

            # ── Key handling ──────────────────────────────────
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == 81 or key == ord('a'):   # LEFT arrow
                phrase_idx  = (phrase_idx - 1) % len(PHRASES)
                capturing   = False
                frame_buffer.clear()
                last_hand_list = None
                dropout_count  = 0
                print(f"[SELECT] {PHRASES[phrase_idx]}  ({counts.get(PHRASES[phrase_idx], 0)} seqs)")

            elif key == 83:                      # RIGHT arrow only (removed 'd' fallback)
                phrase_idx  = (phrase_idx + 1) % len(PHRASES)
                capturing   = False
                frame_buffer.clear()
                last_hand_list = None
                dropout_count  = 0
                print(f"[SELECT] {PHRASES[phrase_idx]}  ({counts.get(PHRASES[phrase_idx], 0)} seqs)")

            elif key == ord(' '):
                capturing = not capturing
                frame_buffer.clear()
                last_hand_list = None
                dropout_count  = 0
                state = "RECORDING" if capturing else "PAUSED"
                print(f"[{state}] {PHRASES[phrase_idx]}")

            elif key == 8 or key == ord('x'):    # Backspace or X = delete last
                phrase = PHRASES[phrase_idx]
                for i in range(len(rows) - 1, -1, -1):
                    if rows[i]["label"] == phrase:
                        rows.pop(i)
                        counts[phrase] = max(0, counts[phrase] - 1)
                        print(f"[DELETE] Removed 1 sequence for '{phrase}'")
                        break

            elif key == ord('s'):
                print("\n── Sequence counts ──────────────────────")
                total = 0
                for p in PHRASES:
                    c    = counts.get(p, 0)
                    icon = "✌" if p in TWO_HAND_PHRASES else " "
                    bar  = "█" * (c // 5) + "░" * max(0, (SAMPLES_GOAL - c) // 5)
                    ok   = "✓" if c >= SAMPLES_GOAL else " "
                    print(f"  {ok}{icon} {p:<25} {c:>4}  {bar}")
                    total += c
                print(f"  Total sequences: {total}\n")

            # Jump to phrase by number key 1-9, 0=10
            elif ord('1') <= key <= ord('9'):
                phrase_idx = key - ord('1')
                capturing  = False
                frame_buffer.clear()
                last_hand_list = None
                dropout_count  = 0
                print(f"[SELECT] {PHRASES[phrase_idx]}")

            elif key == ord('0'):
                phrase_idx = 9
                capturing  = False
                frame_buffer.clear()
                last_hand_list = None
                dropout_count  = 0
                print(f"[SELECT] {PHRASES[phrase_idx]}")

    cap.release()
    cv2.destroyAllWindows()

    save_all(OUTPUT_FILE, rows)
    print(f"\n[SAVED] {len(rows)} total sequences → {OUTPUT_FILE}")
    print("── Final counts ─────────────────────────")
    counts = count_per_phrase(rows)
    for p in PHRASES:
        status = "✓" if counts[p] >= SAMPLES_GOAL else "✗"
        icon   = "✌" if p in TWO_HAND_PHRASES else " "
        print(f"  {status}{icon} {p:<25} {counts[p]}")
    print("\nRun  python train_phrases.py  to retrain the model.")


if __name__ == "__main__":
    main()