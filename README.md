# Audio pipeline & voiceprint verification (Member C)

| File | Purpose |
|---|---|
| `audio_pipeline.py` | Load, clean, visualise, augment, extract features → `data/audio_features.csv` |
| `voice_model.py` | Train/evaluate the voiceprint model, save `models/voice_model.joblib`, `verify()` for the CLI app |
| `audio_processing.ipynb` | Notebook with plots, interpretation and metrics (for the report) |
| `figures/` | Waveforms/spectrograms, augmentations, feature box plots, confusion matrix (generated) |

## 1. Add the recordings

Put each member's recordings, in any format (m4a, aac, mp4, mp3, wav), in a folder named after them:

```
data/audio/original/Emma Tiffany Umwari/yes_approve.m4a
data/audio/original/Emma Tiffany Umwari/confirm_transaction.m4a
data/audio/original/Emma Tiffany Umwari/yes_approve_2.m4a        <- extra takes: any name with "approve"/"confirm"
...
data/audio/unauthorized/stranger_1.m4a                            <- people outside the team
```

`prepare_audio.py` converts them to `data/audio/raw/<firstname>_<phrase>_<take>.wav` (e.g. `emma_yes_approve_1.wav`). The member ID is the **first name in lowercase**, so the face pipeline (Member B) must use the same IDs.

## 2. Run

```bash
pip install -r requirements.txt
python prepare_audio.py           # convert + rename recordings
python audio_pipeline.py          # -> data/audio_features.csv + figures/
python voice_model.py train       # -> models/voice_model.joblib + metrics json
python voice_model.py verify data/audio/raw/emma_yes_approve_1.wav --expected emma
```

Or open `audio_processing.ipynb` and run all cells.

## Method

- **Pre-processing**: 16 kHz mono, trim silence (30 dB), peak-normalise.
- **Augmentations (5 per clip)**: pitch ±2 semitones, time stretch ×1.15 / ×0.85, background noise at 15 dB SNR combined with 0.6× gain and a 100 ms shift.
- **Features (64)**: mean and std of 13 MFCCs, 13 delta-MFCCs, spectral roll-off, centroid, bandwidth, zero-crossing rate, RMS energy and pitch (YIN).
- **Model**: Logistic Regression vs Random Forest, compared with leave-one-recording-out cross-validation. Augmented copies are always held out with their source recording, so there is no leakage. The model with the lowest log loss is kept. Reported metrics are accuracy, macro F1 and log loss.
- **Unknown-speaker rejection**: a clip is approved only if
  1. the classifier's confidence is at least `prob_threshold`, **and**
  2. the clip is close enough to that member's average voiceprint (`dist_threshold`), **and**
  3. the voice belongs to the same member the face model recognised (`expected_member`).

  Both thresholds are learned from held-out predictions.

## For Member D (CLI integration)

```python
from voice_model import VoiceVerifier

verifier = VoiceVerifier.load()                    # loads models/voice_model.joblib
result = verifier.verify("clip.wav", expected_member=face_member)
result["approved"]          # True / False
result["reason"]            # e.g. "voice verified", "low confidence (0.41 < 0.70)", ...
result["predicted_member"], result["confidence"]
```
