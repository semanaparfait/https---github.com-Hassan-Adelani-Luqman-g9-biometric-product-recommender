"""
Audio pipeline (Member C) - load, clean, visualise, augment and extract features
from each team member's voice recordings, then write audio_features.csv.

Expected input layout (file names drive the labels):

    data/audio/raw/<member>_<phrase>_<take>.wav
        e.g.  alice_yes_approve_1.wav
              alice_confirm_transaction_1.wav
    data/audio/unauthorized/<anything>.wav      (non-members, used only for the demo/eval)

Run:  python audio_pipeline.py            -> writes data/audio_features.csv + figures/
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# librosa 1.0 warns about its own internal time_stretch args; yin warns on silent frames
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

# ----------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "audio" / "raw"
UNAUTH_DIR = ROOT / "data" / "audio" / "unauthorized"
FEATURES_CSV = ROOT / "data" / "audio_features.csv"
FIG_DIR = ROOT / "figures"

SR = 16_000          # every clip is resampled to 16 kHz mono
N_MFCC = 13
TOP_DB = 30          # silence-trim threshold
AUDIO_EXTS = {".wav"}   # prepare_audio.py converts everything else to wav
FILENAME_RE = re.compile(r"^(?P<member>[A-Za-z0-9]+)_(?P<phrase>[A-Za-z_]+?)_(?P<take>\d+)$")


# ----------------------------------------------------------------------------- loading
def parse_filename(path: Path) -> dict:
    """alice_yes_approve_1.wav -> {member: alice, phrase: yes_approve, take: 1}"""
    m = FILENAME_RE.match(path.stem)
    if not m:
        raise ValueError(
            f"Bad file name '{path.name}'. Use <member>_<phrase>_<take>.wav, e.g. alice_yes_approve_1.wav"
        )
    return {"member": m["member"].lower(), "phrase": m["phrase"].lower(), "take": int(m["take"])}


def list_audio(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXTS)


def load_clip(path: str | Path, sr: int = SR) -> np.ndarray:
    """Load as mono, resample, trim leading/trailing silence, peak-normalise."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"{path} is empty")
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    return peak_normalize(y)


def peak_normalize(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def load_dataset(folder: Path = RAW_DIR) -> pd.DataFrame:
    """One row per original recording: path, member, phrase, take, signal, duration."""
    rows = []
    for p in list_audio(folder):
        y = load_clip(p)
        rows.append({"file": p.name, "path": str(p), **parse_filename(p),
                     "signal": y, "duration_s": len(y) / SR})
    if not rows:
        raise FileNotFoundError(f"No audio found in {folder}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- augmentation
def add_noise(y: np.ndarray, snr_db: float = 20.0, rng=None) -> np.ndarray:
    """Add white background noise at a given signal-to-noise ratio."""
    rng = rng or np.random.default_rng(0)
    noise_power = np.mean(y ** 2) / (10 ** (snr_db / 10))
    return y + rng.normal(0, np.sqrt(noise_power), y.shape)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Pad to a whole number of STFT hops first: librosa 1.0's phase vocoder raises
    't_out values must be in the range' for some clip lengths."""
    return librosa.effects.time_stretch(np.pad(y, (0, (-len(y)) % 512)), rate=rate)


def augment(y: np.ndarray, sr: int = SR, seed: int = 0) -> dict[str, np.ndarray]:
    """Five augmentations per clip (rubric asks for >= 2)."""
    rng = np.random.default_rng(seed)
    shift = int(0.1 * sr)
    return {
        "pitch_up":   peak_normalize(librosa.effects.pitch_shift(y, sr=sr, n_steps=2)),
        "pitch_down": peak_normalize(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)),
        "stretch_fast": peak_normalize(time_stretch(y, 1.15)),
        "stretch_slow": peak_normalize(time_stretch(y, 0.85)),
        # quieter + delayed + noisy: simulates a worse microphone / room
        "noise_gain_shift": add_noise(np.concatenate([np.zeros(shift), 0.6 * y]), snr_db=15, rng=rng),
    }


# ----------------------------------------------------------------------------- features
def _stats(name: str, m: np.ndarray) -> dict[str, float]:
    """Mean and std over time for each row of a (n_coeffs, frames) matrix."""
    m = np.atleast_2d(m)
    out = {}
    for i, row in enumerate(m):
        suffix = f"{name}_{i + 1}" if m.shape[0] > 1 else name
        out[f"{suffix}_mean"] = float(np.mean(row))
        out[f"{suffix}_std"] = float(np.std(row))
    return out


def extract_features(y: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Fixed-length feature vector for one clip (64 values). Duration is left out on purpose:
    it describes the phrase, not the speaker.."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feats = {}
    feats |= _stats("mfcc", mfcc)                                   # timbre / vocal tract shape
    feats |= _stats("mfcc_delta", librosa.feature.delta(mfcc))      # how timbre changes
    feats |= _stats("rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    feats |= _stats("centroid", librosa.feature.spectral_centroid(y=y, sr=sr))
    feats |= _stats("bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr))
    feats |= _stats("zcr", librosa.feature.zero_crossing_rate(y))
    feats |= _stats("rms_energy", librosa.feature.rms(y=y))
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)                   # pitch of the speaker
    feats |= _stats("pitch_f0", f0)
    return feats


def feature_columns(df: pd.DataFrame) -> list[str]:
    meta = {"file", "member", "phrase", "take", "augmentation", "source_file", "path"}
    return [c for c in df.columns if c not in meta]


def build_feature_table(dataset: pd.DataFrame) -> pd.DataFrame:
    """Original + augmented versions of every clip -> one feature row each.
    `source_file` groups augmented copies with their original so the model can be
    evaluated without leakage."""
    rows = []
    for i, r in dataset.iterrows():
        versions = {"original": r.signal} | augment(r.signal, seed=i)
        for aug_name, y in versions.items():
            rows.append({"file": r.file, "member": r.member, "phrase": r.phrase, "take": r["take"],
                         "augmentation": aug_name, "source_file": r.file, **extract_features(y)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- plots
def plot_waveform_and_spectrogram(y: np.ndarray, title: str, sr: int = SR, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set(title=f"{title} - waveform", xlabel="Time (s)", ylabel="Amplitude")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=128)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, hop_length=128, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set(title=f"{title} - spectrogram (dB)", ylim=(0, 5000))
    plt.colorbar(img, ax=axes[1], format="%+2.0f dB")
    return axes


def plot_all_members(dataset: pd.DataFrame, save_to: Path | None = None):
    """One row per recording: waveform + spectrogram."""
    fig, axes = plt.subplots(len(dataset), 2, figsize=(13, 2.8 * len(dataset)), squeeze=False)
    for ax_row, (_, r) in zip(axes, dataset.iterrows()):
        plot_waveform_and_spectrogram(r.signal, f"{r.member} / {r.phrase} #{r['take']}", axes=ax_row)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=110)
    return fig


def plot_augmentations(y: np.ndarray, title: str, save_to: Path | None = None):
    versions = {"original": y} | augment(y)
    fig, axes = plt.subplots(len(versions), 2, figsize=(13, 2.6 * len(versions)))
    for ax_row, (name, sig) in zip(axes, versions.items()):
        plot_waveform_and_spectrogram(sig, f"{title} - {name}", axes=ax_row)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=110)
    return fig


# ----------------------------------------------------------------------------- main
def main():
    FIG_DIR.mkdir(exist_ok=True)
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} recordings from {dataset.member.nunique()} members")
    print(dataset.groupby("member").agg(clips=("file", "count"), phrases=("phrase", "nunique"),
                                        mean_duration_s=("duration_s", "mean")).round(2))

    plot_all_members(dataset, FIG_DIR / "audio_waveforms_spectrograms.png")
    first = dataset.iloc[0]
    plot_augmentations(first.signal, first.file, FIG_DIR / "audio_augmentations.png")
    plt.close("all")

    table = build_feature_table(dataset)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FEATURES_CSV, index=False)
    print(f"Wrote {FEATURES_CSV.relative_to(ROOT)}: {table.shape[0]} rows x "
          f"{len(feature_columns(table))} features")


if __name__ == "__main__":
    main()
