"""
train_numbers.py
────────────────
Trains (or retrains) fsl_number_model.joblib from the CSV produced by
collect_numbers.py.

Usage:
    python train_numbers.py

Output:
    fsl_number_model.joblib  (drop-in replacement — fsl_worker.py loads it automatically)
    data/label_report.txt    (per-digit precision / recall)
"""

import os
import csv
import sys

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────
# Config
# ─────────────────────────────
DATA_FILE    = os.path.join("data", "numbers_dataset.csv")
MODEL_OUTPUT = "fsl_number_model.joblib"
REPORT_FILE  = os.path.join("data", "label_report.txt")
MIN_SAMPLES  = 30    # warn if any digit has fewer than this
DIGITS       = [str(d) for d in range(10)]


# ─────────────────────────────
# Load data
# ─────────────────────────────
def load_dataset(path):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Dataset not found: {path}\n"
                 f"        Run  python collect_numbers.py  first.")

    X, y = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if label not in DIGITS:
                continue   # skip any stray rows
            features = [float(row[f"f{i}"]) for i in range(63)]
            X.append(features)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


# ─────────────────────────────
# Augmentation (mild jitter)
# ─────────────────────────────
def augment(X, y, factor=3, noise_std=0.008):
    """
    Synthetically multiply the dataset by adding tiny Gaussian noise.
    Helps when some digits have fewer samples than others.
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
    Voting ensemble of three fast learners.
    RandomForest is the workhorse; SVC adds a different decision boundary;
    GradientBoosting handles hard inter-class confusions.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    svc = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=42)),
    ])
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
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
    print("=" * 52)
    print("  FSL Number Model Trainer  (digits 0-9)")
    print("=" * 52)

    # 1. Load
    print(f"\n[1/5] Loading dataset from  {DATA_FILE} …")
    X, y = load_dataset(DATA_FILE)
    print(f"      {len(X)} total samples loaded.")

    # 2. Per-digit count & warnings
    print("\n[2/5] Sample distribution:")
    labels, cnts = np.unique(y, return_counts=True)
    all_ok = True
    for d in DIGITS:
        if d in labels:
            c = cnts[list(labels).index(d)]
            warn = "  ⚠ LOW" if c < MIN_SAMPLES else ""
            bar  = "▓" * (c // 10) + "░" * max(0, (MIN_SAMPLES - c) // 10)
            print(f"      Digit {d}: {c:>4} {bar}{warn}")
            if c < MIN_SAMPLES:
                all_ok = False
        else:
            print(f"      Digit {d}:    0  ← MISSING — collect samples first")
            all_ok = False

    if not all_ok:
        ans = input("\n  Some digits have very few samples. Continue anyway? [y/N] ").strip().lower()
        if ans != "y":
            print("  Aborted. Collect more samples and try again.")
            return

    # 3. Augment
    print("\n[3/5] Augmenting dataset (×3 with jitter) …")
    X_aug, y_aug = augment(X, y, factor=3)
    print(f"      {len(X_aug)} samples after augmentation.")

    # 4. Train / evaluate
    print("\n[4/5] Training ensemble model …")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug
    )

    clf = build_model()
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    print(f"\n      Hold-out accuracy : {acc * 100:.1f}%")

    # Cross-validation on original (non-augmented) data
    cv_scores = cross_val_score(
        build_model(), X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy", n_jobs=-1
    )
    print(f"      5-fold CV accuracy: {cv_scores.mean() * 100:.1f}% ± {cv_scores.std() * 100:.1f}%")

    # Per-class report
    report = classification_report(y_te, clf.predict(X_te), target_names=sorted(set(y)))
    print("\n── Per-digit precision / recall ──────────────")
    print(report)

    cm = confusion_matrix(y_te, clf.predict(X_te), labels=sorted(set(y)))
    print("── Confusion matrix (rows=true, cols=pred) ──")
    print("   " + "  ".join(sorted(set(y))))
    for lbl, row in zip(sorted(set(y)), cm):
        print(f" {lbl} " + "  ".join(f"{v:3}" for v in row))

    # Save report
    os.makedirs("data", exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"\n      Report saved → {REPORT_FILE}")

    # 5. Save model
    print(f"\n[5/5] Saving model → {MODEL_OUTPUT} …")
    joblib.dump(clf, MODEL_OUTPUT)
    size_kb = os.path.getsize(MODEL_OUTPUT) / 1024
    print(f"      Done! ({size_kb:.0f} KB)")

    print("\n" + "=" * 52)
    print("  Training complete.")
    print(f"  Drop  {MODEL_OUTPUT}  next to  fsl_worker.py")
    print("  and restart the Flask app — no other changes needed.")
    print("=" * 52)


if __name__ == "__main__":
    main()