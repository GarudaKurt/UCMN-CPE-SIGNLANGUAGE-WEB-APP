"""
collect_numbers.py
──────────────────
Interactive data collector for FSL Numbers 0–9.

Usage:
    python collect_numbers.py

Controls (while the camera window is open):
    0-9       → select which digit to collect
    SPACE     → start / pause capture for the current digit
    D         → delete last saved sample for the current digit
    S         → show current sample counts
    Q / ESC   → quit and save everything to  data/numbers_dataset.csv

Output:
    data/numbers_dataset.csv  – one row per sample, columns:
        label, f0, f1, ..., f62   (label + 63 normalised landmark features)
"""

import os
import csv
import time
import urllib.request

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
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "numbers_dataset.csv")
CAMERA_INDEX = 0

DIGITS       = [str(d) for d in range(10)]   # "0" … "9"
SAMPLES_GOAL = 200   # target samples per digit (shown in UI)

# Colours (BGR)
C_GREEN  = (0,  220,  80)
C_RED    = (0,   40, 220)
C_YELLOW = (0,  200, 200)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,     0,   0)
C_BLUE   = (220,  80,   0)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ─────────────────────────────
# Landmark normalisation
# (must match fsl_worker.py exactly)
# ─────────────────────────────
def normalize_landmarks(hand_landmarks):
    wrist = hand_landmarks[0]
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    features = []
    for lm in hand_landmarks:
        features.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale,
        ])
    return features   # 21 landmarks × 3 = 63 floats


# ─────────────────────────────
# Drawing helpers
# ─────────────────────────────
def draw_hand(frame, hand_landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, C_GREEN, -1)
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], C_BLUE, 2)


def put_text(frame, text, pos, scale=0.8, color=C_WHITE, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, C_BLACK, thickness + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thickness)


def draw_hud(frame, digit, capturing, counts, hand_detected, last_saved_time):
    h, w = frame.shape[:2]

    # ── top bar ──────────────────────────────────────────────
    bar_h = 56
    cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)

    status_color = C_GREEN if capturing else C_YELLOW
    status_text  = "● REC" if capturing else "  PAUSED"
    put_text(frame, status_text, (12, 36), 0.9, status_color)

    digit_label = f"Digit: {digit}" if digit else "Digit: — (press 0-9)"
    put_text(frame, digit_label, (160, 36), 0.9, C_WHITE)

    count = counts.get(digit, 0) if digit else 0
    bar_w = int(min(count / SAMPLES_GOAL, 1.0) * 220)
    cv2.rectangle(frame, (w - 270, 14), (w - 50, 42), (60, 60, 60), -1)
    bar_color = C_GREEN if count >= SAMPLES_GOAL else C_YELLOW
    if bar_w > 0:
        cv2.rectangle(frame, (w - 270, 14), (w - 270 + bar_w, 42), bar_color, -1)
    put_text(frame, f"{count}/{SAMPLES_GOAL}", (w - 45, 36), 0.7, C_WHITE, 1)

    # ── hand status ─────────────────────────────────────────
    hc = C_GREEN if hand_detected else C_RED
    ht = "Hand: detected" if hand_detected else "Hand: not found"
    put_text(frame, ht, (12, h - 60), 0.65, hc, 1)

    # ── flash on save ────────────────────────────────────────
    if last_saved_time and (time.time() - last_saved_time) < 0.3:
        cv2.rectangle(frame, (0, 0), (w, h), C_GREEN, 6)
        put_text(frame, "SAVED", (w // 2 - 50, h // 2), 1.5, C_GREEN, 3)

    # ── controls legend ──────────────────────────────────────
    legend = [
        "0-9 → select digit",
        "SPACE → rec on/off",
        "D → delete last",
        "S → show counts",
        "Q/ESC → quit & save",
    ]
    for i, txt in enumerate(legend):
        put_text(frame, txt, (12, h - 110 - i * 22), 0.50, C_WHITE, 1)

    # ── digit grid at bottom-right ───────────────────────────
    grid_x, grid_y = w - 220, h - 115
    for i, d in enumerate(DIGITS):
        col = i % 5
        row = i // 5
        bx  = grid_x + col * 42
        by  = grid_y + row * 42
        c   = counts.get(d, 0)
        bg  = C_GREEN if c >= SAMPLES_GOAL else (80, 80, 80)
        if d == digit:
            bg = C_YELLOW
        cv2.rectangle(frame, (bx, by), (bx + 36, by + 36), bg, -1)
        put_text(frame, d, (bx + 10, by + 26), 0.7, C_BLACK if bg != (80,80,80) else C_WHITE, 1)
        put_text(frame, str(c), (bx + 2, by + 36 + 14), 0.40, C_WHITE, 1)


# ─────────────────────────────
# CSV helpers
# ─────────────────────────────
HEADER = ["label"] + [f"f{i}" for i in range(63)]

def load_existing(path):
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def save_all(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def count_per_digit(rows):
    counts = {d: 0 for d in DIGITS}
    for row in rows:
        lbl = row.get("label", "")
        if lbl in counts:
            counts[lbl] += 1
    return counts


# ─────────────────────────────
# Main
# ─────────────────────────────
def main():
    # Download model if needed
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded.")

    # MediaPipe setup
    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )

    # Load existing data
    rows      = load_existing(OUTPUT_FILE)
    counts    = count_per_digit(rows)
    print(f"[INFO] Loaded {len(rows)} existing samples.")
    for d in DIGITS:
        print(f"  Digit {d}: {counts[d]} samples")

    # State
    current_digit   = None
    capturing       = False
    last_saved_time = None

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("\n[INFO] Camera open. Press 0-9 to select a digit, SPACE to record, Q to quit.\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)   # mirror
            h, w  = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            hand_lms = result.hand_landmarks[0] if result.hand_landmarks else None

            # ── capture sample ────────────────────────────────
            if capturing and current_digit and hand_lms:
                features = normalize_landmarks(hand_lms)
                row = {"label": current_digit}
                row.update({f"f{i}": round(v, 6) for i, v in enumerate(features)})
                rows.append(row)
                counts[current_digit] = counts.get(current_digit, 0) + 1
                last_saved_time = time.time()

            # ── draw ──────────────────────────────────────────
            if hand_lms:
                draw_hand(frame, hand_lms, w, h)

            draw_hud(frame, current_digit, capturing, counts, hand_lms is not None, last_saved_time)

            cv2.imshow("FSL Number Collector  |  q = quit", frame)

            # ── key handling ──────────────────────────────────
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):   # Q or ESC
                break

            elif chr(key) in DIGITS if key < 128 else False:
                new_digit = chr(key)
                if new_digit != current_digit:
                    current_digit = new_digit
                    capturing     = False
                    print(f"[SELECT] Digit {current_digit}  ({counts.get(current_digit, 0)} samples so far)")

            elif key == ord(' '):
                if current_digit is None:
                    print("[WARN] Select a digit first (press 0-9).")
                else:
                    capturing = not capturing
                    state = "RECORDING" if capturing else "PAUSED"
                    print(f"[{state}] Digit {current_digit}")

            elif key == ord('d'):
                # Delete last saved sample for current digit
                if current_digit and rows:
                    for i in range(len(rows) - 1, -1, -1):
                        if rows[i]["label"] == current_digit:
                            rows.pop(i)
                            counts[current_digit] = max(0, counts[current_digit] - 1)
                            print(f"[DELETE] Removed 1 sample for digit {current_digit}.")
                            break

            elif key == ord('s'):
                print("\n── Sample counts ────────────────")
                total = 0
                for d in DIGITS:
                    c = counts.get(d, 0)
                    bar = "█" * (c // 10)
                    status = "✓" if c >= SAMPLES_GOAL else " "
                    print(f"  {status} Digit {d}: {c:>4}  {bar}")
                    total += c
                print(f"  Total: {total}\n")

            # ── throttle to ~15 fps capture ────────────────────
            time.sleep(0.067)

    cap.release()
    cv2.destroyAllWindows()

    # Save
    save_all(OUTPUT_FILE, rows)
    total = len(rows)
    print(f"\n[SAVED] {total} samples → {OUTPUT_FILE}")
    print("── Final counts ─────────────────")
    counts = count_per_digit(rows)
    for d in DIGITS:
        status = "✓" if counts[d] >= SAMPLES_GOAL else "✗"
        print(f"  {status} Digit {d}: {counts[d]}")
    print("\nRun  python train_numbers.py  to retrain the model.")


if __name__ == "__main__":
    main()