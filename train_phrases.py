"""
train_phrases.py
────────────────
Trains (or retrains) fsl_phrase_model.joblib from the CSV produced by
collect_phrases.py.

TWO-HAND UPDATE
───────────────
NUM_FEATURES is now 3780 (30 frames × 2 hands × 21 landmarks × 3 coords).
The model architecture is unchanged — the ensemble is feature-size agnostic.
Simply re-collect all phrases with collect_phrases.py, then run this script.

Usage:
    python train_phrases.py

Output:
    fsl_phrase_model.joblib   (drop-in replacement loaded by fsl_worker.py)
    data/phrase_report.txt    (per-phrase precision / recall)
"""

import os
import csv
import sys

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# ─────────────────────────────
# Config
# ─────────────────────────────
DATA_FILE    = os.path.join("data", "phrases_dataset.csv")
MODEL_OUTPUT = "fsl_phrase_model.joblib"
REPORT_FILE  = os.path.join("data", "phrase_report.txt")
MIN_SAMPLES  = 20

SEQUENCE_LENGTH = 30
NUM_HANDS       = 2
NUM_FEATURES    = SEQUENCE_LENGTH * NUM_HANDS * 21 * 3   # 3780 (was 1890 with 1 hand)

PHRASES = [
    "Hello", "Thank you", "Please", "Sorry", "Yes",
    "No", "Help me", "I love you", "Good morning", "Good night",
    "How are you", "I am fine", "What is your name", "My name is",
    "Nice to meet you", "Goodbye", "Come here", "Wait",
    "I don't understand", "Can you repeat", "Eat", "Drink",
    "Bathroom", "I need help", "You are welcome",
]

# Known two-hand signs — printed in the distribution report
TWO_HAND_PHRASES = {"How are you", "I love you", "Nice to meet you", "You are welcome"}


# ─────────────────────────────
# Load data
# ─────────────────────────────
def load_dataset(path):
    if not os.path.exists(path):
        sys.exit(
            f"[ERROR] Dataset not found: {path}\n"
            f"        Run  python collect_phrases.py  first."
        )

    X, y = [], []
    old_feature_count = SEQUENCE_LENGTH * 21 * 3   # 1890 — old 1-hand format

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header_features = len(reader.fieldnames or []) - 1   # subtract 'label' column

        if header_features == old_feature_count:
            sys.exit(
                f"[ERROR] Dataset was collected with the OLD 1-hand format "
                f"({old_feature_count} features).\n"
                f"        Please re-collect all phrases using the updated "
                f"collect_phrases.py (2-hand, {NUM_FEATURES} features)."
            )

        if header_features != NUM_FEATURES:
            sys.exit(
                f"[ERROR] Dataset has {header_features} features but "
                f"{NUM_FEATURES} are expected.\n"
                f"        Re-collect with the current collect_phrases.py."
            )

        for row in reader:
            label = row.get("label", "")
            if label not in PHRASES:
                continue
            try:
                features = [float(row[f"f{i}"]) for i in range(NUM_FEATURES)]
            except (KeyError, ValueError):
                continue
            X.append(features)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


# ─────────────────────────────
# Augmentation
# ─────────────────────────────
def augment(X, y, factor=4, noise_std=0.005):
    """
    Add small Gaussian noise to simulate natural variation in speed/angle.
    Zero-padded regions (absent hand) stay near zero — noise is negligible there.
    """
    Xa, ya = [X], [y]
    rng = np.random.default_rng(42)
    for _ in range(factor - 1):
        noise = rng.normal(0, noise_std, X.shape).astype(np.float32)
        Xa.append(X + noise)
        ya.append(y)
    return np.vstack(Xa), np.concatenate(ya)


# ─────────────────────────────
# Build classifier
# ─────────────────────────────
def build_model():
    """
    Ensemble of RF + SVC + GradientBoosting with soft voting.
    The larger feature space (3780) is handled well by this combination:
    - RF uses sqrt(3780) ≈ 61 features per split — still fast
    - SVC with RBF + StandardScaler handles the larger input cleanly
    - GB adds fine-grained inter-phrase boundary tuning
    """
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    svc = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf", C=20, gamma="scale",
            probability=True, random_state=42,
        )),
    ])
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.07,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("svc", svc), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )


# ─────────────────────────────
# Main
# ─────────────────────────────
def main():
    print("=" * 56)
    print("  FSL Phrase Model Trainer  (25 phrases, 2-hand)")
    print("=" * 56)
    print(f"\n  Feature size: {NUM_FEATURES} "
          f"({SEQUENCE_LENGTH} frames × {NUM_HANDS} hands × 21 lm × 3 coords)")

    # 1. Load
    print(f"\n[1/5] Loading dataset from  {DATA_FILE} …")
    X, y = load_dataset(DATA_FILE)
    print(f"      {len(X)} total sequences loaded.")

    if len(X) == 0:
        sys.exit("[ERROR] No data found. Run collect_phrases.py first.")

    # 2. Distribution & warnings
    print("\n[2/5] Sequence distribution:")
    labels_found = sorted(set(y))
    all_ok = True
    for p in PHRASES:
        c    = int(np.sum(y == p))
        icon = "✌" if p in TWO_HAND_PHRASES else " "
        warn = "  ⚠ LOW" if c < MIN_SAMPLES else ""
        bar  = "▓" * (c // 5) + "░" * max(0, (MIN_SAMPLES - c) // 5)
        status = "✓" if c >= MIN_SAMPLES else "✗"
        print(f"      {status}{icon} {p:<25} {c:>4}  {bar}{warn}")
        if c < MIN_SAMPLES:
            all_ok = False

    missing = [p for p in PHRASES if p not in labels_found]
    if missing:
        print(f"\n  ⚠ Missing phrases (no data collected):")
        for m in missing:
            icon = "✌" if m in TWO_HAND_PHRASES else " "
            print(f"    -{icon} {m}")
        all_ok = False

    if not all_ok:
        ans = input(
            "\n  Some phrases have very few or no sequences. Continue anyway? [y/N] "
        ).strip().lower()
        if ans != "y":
            print("  Aborted. Collect more sequences and try again.")
            return

    # 3. Augment
    print(f"\n[3/5] Augmenting dataset (×4 with jitter) …")
    X_aug, y_aug = augment(X, y, factor=4)
    print(f"      {len(X_aug)} sequences after augmentation.")

    # 4. Train / evaluate
    print("\n[4/5] Training ensemble model …")
    print("      (With 3780 features this may take 2-5 min — please wait)\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug
    )

    clf = build_model()
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    print(f"\n      Hold-out accuracy:  {acc * 100:.1f}%")

    # Cross-validation on original (non-augmented) data to avoid leakage
    print("      Running 5-fold CV on original data …")
    label_indices = np.array([labels_found.index(lbl) for lbl in y])
    min_class_count = int(np.bincount(label_indices).min())
    n_splits = min(5, min_class_count)
    if n_splits >= 2:
        cv_scores = cross_val_score(
            build_model(), X, y,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
            scoring="accuracy", n_jobs=-1,
        )
        print(f"      {n_splits}-fold CV accuracy: {cv_scores.mean() * 100:.1f}% ± {cv_scores.std() * 100:.1f}%")
    else:
        print("      Skipping CV — not enough samples per class.")

    # Per-phrase report
    y_pred  = clf.predict(X_te)
    report  = classification_report(y_te, y_pred, target_names=sorted(set(y)))
    print("\n── Per-phrase precision / recall ──────────────────────")
    print(report)

    # Confusion matrix — show confused pairs only
    cm            = confusion_matrix(y_te, y_pred, labels=sorted(set(y)))
    sorted_labels = sorted(set(y))
    print("── Commonly confused pairs ────────────────────────────")
    found_any = False
    for i, true_lbl in enumerate(sorted_labels):
        for j, pred_lbl in enumerate(sorted_labels):
            if i != j and cm[i][j] > 0:
                print(f"   '{true_lbl}' predicted as '{pred_lbl}': {cm[i][j]}x")
                found_any = True
    if not found_any:
        print("   (none — perfect separation on test set)")

    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(f"Feature size: {NUM_FEATURES} (2-hand)\n\n")
        f.write(report)
    print(f"\n      Report saved → {REPORT_FILE}")

    # 5. Save model
    print(f"\n[5/5] Saving model → {MODEL_OUTPUT} …")
    joblib.dump(clf, MODEL_OUTPUT)
    size_kb = os.path.getsize(MODEL_OUTPUT) / 1024
    print(f"      Done! ({size_kb:.0f} KB)\n")

    print("=" * 56)
    print("  Training complete.")
    print(f"  Drop  {MODEL_OUTPUT}  next to  fsl_worker.py")
    print("  and restart the Flask app — no other changes needed.")
    print("=" * 56)


if __name__ == "__main__":
    main()