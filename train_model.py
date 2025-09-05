# train_model.py
# Creates: models/best_random_forest_model.pkl and models/scaler.pkl
# Matches the feature order & encodings used by your Streamlit app.

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------- Config -------------------- #
np.random.seed(42)

FEATURE_NAMES = [
    "Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly",
    "Absences", "Tutoring", "ParentalSupport", "Extracurricular",
    "Sports", "Music", "Volunteering"
]

# Encoders (must mirror your app!)
GENDER_MAP = {"Male": 1, "Female": 0}
ETHNICITY_MAP = {"Group A": 0, "Group B": 1, "Group C": 2, "Group D": 3, "Group E": 4}
EDU_MAP = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
SUPPORT_MAP = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3, "Excellent": 4}
YESNO_MAP = {"Yes": 1, "No": 0}

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "best_random_forest_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# ---------------- Synthetic data generator ---------------- #
def synthesize_dataset(n=1200):
    # Categorical raw values
    genders = np.random.choice(list(GENDER_MAP.keys()), size=n, p=[0.5, 0.5])
    ethnicities = np.random.choice(list(ETHNICITY_MAP.keys()), size=n)
    edu = np.random.choice(list(EDU_MAP.keys()), size=n, p=[0.18, 0.20, 0.32, 0.22, 0.08])
    support = np.random.choice(list(SUPPORT_MAP.keys()), size=n, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    tutoring = np.random.choice(list(YESNO_MAP.keys()), size=n, p=[0.35, 0.65])
    extracurricular = np.random.choice(list(YESNO_MAP.keys()), size=n, p=[0.55, 0.45])

    # Numeric features (simple plausible ranges)
    age = np.random.randint(15, 26, size=n)                         # 15–25
    study = np.clip(np.round(np.random.normal(14, 6, size=n)), 0, 40).astype(int)
    absences = np.clip(np.round(np.random.normal(6, 6, size=n)), 0, 40).astype(int)
    sports = np.clip(np.round(np.random.normal(4, 3, size=n)), 0, 20).astype(int)
    music = np.clip(np.round(np.random.normal(1, 2, size=n)), 0, 12).astype(int)
    volunteering = np.clip(np.round(np.random.normal(0.5, 1.5, size=n)), 0, 10).astype(int)

    # Encode categoricals to match the app
    gender_enc = np.vectorize(GENDER_MAP.get)(genders)
    eth_enc = np.vectorize(ETHNICITY_MAP.get)(ethnicities)
    edu_enc = np.vectorize(EDU_MAP.get)(edu)
    sup_enc = np.vectorize(SUPPORT_MAP.get)(support)
    tut_enc = np.vectorize(YESNO_MAP.get)(tutoring)
    extra_enc = np.vectorize(YESNO_MAP.get)(extracurricular)

    # Build a latent score -> label (so model has signal to learn)
    # Normalize some features for scoring
    study_norm = (study / 40.0)        # 0..1
    abs_norm = (absences / 40.0)       # 0..1 (higher is worse)
    sports_norm = (sports / 20.0)      # 0..1 (too high can hurt)
    music_norm = (music / 12.0)        # 0..1
    vol_norm = (volunteering / 10.0)   # 0..1

    score = (
        0.50 * study_norm
        - 0.40 * abs_norm
        + 0.18 * (sup_enc / 4.0)         # 0..1
        + 0.10 * (tut_enc)               # 0 or 1
        + 0.06 * (extra_enc)             # 0 or 1
        + 0.05 * (edu_enc / 4.0)         # 0..1
        - 0.07 * np.maximum(sports_norm - 0.5, 0)   # too much sports hurts
        + 0.03 * music_norm
        + 0.02 * vol_norm
        + np.random.normal(0, 0.05, size=n)         # noise
    )

    # Thresholds for labels
    y = np.where(score >= 0.55, "Passing",
         np.where(score >= 0.35, "Average", "Low"))

    # Assemble DataFrame in the EXACT order your app expects
    X = pd.DataFrame({
        "Age": age,
        "Gender": gender_enc,
        "Ethnicity": eth_enc,
        "ParentalEducation": edu_enc,
        "StudyTimeWeekly": study,
        "Absences": absences,
        "Tutoring": tut_enc,
        "ParentalSupport": sup_enc,
        "Extracurricular": extra_enc,
        "Sports": sports,
        "Music": music,
        "Volunteering": volunteering,
    })[FEATURE_NAMES]

    return X, y

# -------------------- Train & Save -------------------- #
def main():
    print("Generating synthetic dataset...")
    X, y = synthesize_dataset(n=1500)

    # Optional: save a copy for inspection
    pd.DataFrame(X).assign(Label=y).to_csv("training_sample.csv", index=False)
    print("Saved sample dataset to training_sample.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Class weights help if synthetic balance drifts
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    # Quick eval
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", round(acc, 4))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
    print("Classes learned:", clf.classes_)

    # Save artifacts
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")

if __name__ == "__main__":
    main()
