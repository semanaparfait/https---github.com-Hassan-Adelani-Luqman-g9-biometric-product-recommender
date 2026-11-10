"""
Voiceprint verification model (Member C).

Trains a speaker classifier on audio_features.csv and adds an "unknown speaker"
rejection rule so non-members are refused, not just mapped to the nearest member.

    python voice_model.py train                         # train, evaluate, save models/voice_model.joblib
    python voice_model.py verify clip.wav               # who is speaking? approved?
    python voice_model.py verify clip.wav --expected alice   # must be alice (face result)

From the CLI app (Member D):

    from voice_model import VoiceVerifier
    verifier = VoiceVerifier.load()
    result = verifier.verify("clip.wav", expected_member=face_result)
    if result["approved"]: ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, log_loss)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_pipeline import (FEATURES_CSV, FIG_DIR, ROOT, UNAUTH_DIR, extract_features,
                            feature_columns, list_audio, load_clip)

MODEL_PATH = ROOT / "models" / "voice_model.joblib"
METRICS_PATH = ROOT / "models" / "voice_model_metrics.json"
GENUINE_PERCENTILE = 95   # distance threshold keeps 95% of genuine held-out clips


def candidate_models() -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    }


class CentroidDistance:
    """Distance from a clip to a member's average voice, in standardised feature space.
    A non-member is usually far from every centroid even when the classifier is 'confident'."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.classes_ = np.unique(y)
        self.centroids_ = {c: Z[y == c].mean(axis=0) for c in self.classes_}
        return self

    def distance(self, X: np.ndarray, members) -> np.ndarray:
        Z = self.scaler.transform(X)
        return np.array([np.linalg.norm(z - self.centroids_[m]) / np.sqrt(Z.shape[1])
                         for z, m in zip(Z, members)])


# ----------------------------------------------------------------------------- evaluation
def cross_validate(model, X, y, groups, classes):
    """Leave-one-recording-out: the held-out recording AND all its augmented copies are
    unseen during training, so augmentation cannot leak into the score."""
    proba = np.zeros((len(y), len(classes)))
    genuine_dist = np.zeros(len(y))
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        m = clone(model).fit(X[train], y[train])
        p = m.predict_proba(X[test])
        # align columns in case a fold is missing a class
        proba[np.ix_(test, np.searchsorted(classes, m.classes_))] = p
        genuine_dist[test] = CentroidDistance().fit(X[train], y[train]).distance(X[test], y[test])
    return proba, genuine_dist


def metrics_from_proba(y, proba, classes) -> dict:
    pred = classes[proba.argmax(axis=1)]
    proba = np.clip(proba, 1e-6, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1_macro": round(f1_score(y, pred, average="macro"), 4),
        "log_loss": round(log_loss(y, proba, labels=classes), 4),
    }


def train(features_csv: Path = FEATURES_CSV) -> dict:
    df = pd.read_csv(features_csv)
    cols = feature_columns(df)
    X, y, groups = df[cols].to_numpy(), df["member"].to_numpy(), df["source_file"].to_numpy()
    classes = np.unique(y)
    print(f"{len(df)} rows ({df.source_file.nunique()} original recordings), "
          f"{len(cols)} features, members: {', '.join(classes)}")
    if df.groupby("member").source_file.nunique().min() < 2:
        raise ValueError("Every member needs at least 2 recordings for leave-one-recording-out CV.")

    results, oof = {}, {}
    for name, model in candidate_models().items():
        proba, dist = cross_validate(model, X, y, groups, classes)
        results[name] = metrics_from_proba(y, proba, classes)
        oof[name] = (proba, dist)
        print(f"  {name:20s} {results[name]}")

    best = min(results, key=lambda n: (results[n]["log_loss"], -results[n]["f1_macro"]))
    proba, dist = oof[best]
    pred = classes[proba.argmax(axis=1)]
    print(f"Best model: {best}\n")
    print(classification_report(y, pred, zero_division=0))

    # Thresholds learned from held-out (not training) predictions
    correct_conf = proba.max(axis=1)[pred == y]
    prob_threshold = float(np.clip(np.percentile(correct_conf, 10) if len(correct_conf) else 0.5,
                                   1 / len(classes) + 0.1, 0.8))
    dist_threshold = float(np.percentile(dist, GENUINE_PERCENTILE))

    final = clone(candidate_models()[best]).fit(X, y)
    bundle = {
        "model_name": best, "model": final, "distance": CentroidDistance().fit(X, y),
        "feature_columns": cols, "classes": list(classes),
        "prob_threshold": prob_threshold, "dist_threshold": dist_threshold,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=classes, ax=ax, colorbar=False)
    ax.set_title(f"Voiceprint - {best}\n(leave-one-recording-out)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "voice_confusion_matrix.png", dpi=110); plt.close(fig)

    summary = {"cv_scheme": "leave-one-recording-out (augmented copies held out with their source)",
               "models": results, "best_model": best,
               "prob_threshold": round(prob_threshold, 3), "dist_threshold": round(dist_threshold, 3)}

    # Genuine accept rate of the full rule (classifier + both thresholds) on held-out data
    accepted = (pred == y) & (proba.max(axis=1) >= prob_threshold) & (dist <= dist_threshold)
    summary["genuine_accept_rate_cv"] = round(float(accepted.mean()), 4)

    unauth = list_audio(UNAUTH_DIR) if UNAUTH_DIR.exists() else []
    if unauth:
        verifier = VoiceVerifier(bundle)
        rejected = [not verifier.verify(p)["approved"] for p in unauth]
        summary["unauthorized_reject_rate"] = round(float(np.mean(rejected)), 4)
        summary["unauthorized_clips"] = len(unauth)

    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2))
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)}")
    return summary


# ----------------------------------------------------------------------------- inference
class VoiceVerifier:
    def __init__(self, bundle: dict):
        self.b = bundle

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "VoiceVerifier":
        return cls(joblib.load(path))

    def verify(self, audio_path, expected_member: str | None = None) -> dict:
        """Approve only if the voice is a known member (confident AND close to that member's
        voiceprint) and, when given, is the same person the face model recognised."""
        feats = extract_features(load_clip(audio_path))
        x = pd.DataFrame([feats])[self.b["feature_columns"]].to_numpy()
        proba = self.b["model"].predict_proba(x)[0]
        classes = list(self.b["model"].classes_)
        member = classes[int(proba.argmax())]
        conf = float(proba.max())
        dist = float(self.b["distance"].distance(x, [member])[0])

        if conf < self.b["prob_threshold"]:
            reason = f"low confidence ({conf:.2f} < {self.b['prob_threshold']:.2f})"
        elif dist > self.b["dist_threshold"]:
            reason = f"voice does not match any enrolled voiceprint (distance {dist:.2f} > {self.b['dist_threshold']:.2f})"
        elif expected_member and member != expected_member.lower():
            reason = f"voice belongs to '{member}', but face was '{expected_member}'"
        else:
            reason = "voice verified"
        return {"approved": reason == "voice verified", "predicted_member": member,
                "confidence": round(conf, 3), "distance": round(dist, 3), "reason": reason,
                "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    v = sub.add_parser("verify")
    v.add_argument("audio")
    v.add_argument("--expected", help="member recognised by the face model")
    args = ap.parse_args()

    if args.cmd == "train":
        train()
    else:
        r = VoiceVerifier.load().verify(args.audio, args.expected)
        print(("APPROVED" if r["approved"] else "ACCESS DENIED") + f" - {r['reason']}")
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
