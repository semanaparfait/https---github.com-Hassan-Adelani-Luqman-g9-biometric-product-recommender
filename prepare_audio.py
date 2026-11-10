"""
Step 0 - convert the members' phone recordings (m4a / aac / mp4 / ...) into clean,
consistently named WAV files the pipeline can read.

    data/audio/original/<Full Name>/<any name containing "approve" or "confirm">.<ext>
        ->  data/audio/raw/<firstname>_<phrase>_<take>.wav

e.g.  original/Emma Tiffany Umwari/yes_approve.m4a  ->  raw/emma_yes_approve_1.wav

Extra takes are numbered automatically (yes_approve_2.m4a -> emma_yes_approve_2.wav).
Clips in data/audio/unauthorized/ that are not .wav yet are converted next to the original.

Run:  python prepare_audio.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = ROOT / "data" / "audio" / "original"
RAW_DIR = ROOT / "data" / "audio" / "raw"
UNAUTH_DIR = ROOT / "data" / "audio" / "unauthorized"
MEDIA_EXTS = {".m4a", ".aac", ".mp4", ".mp3", ".wav", ".ogg", ".flac", ".opus", ".3gp", ".webm"}


def member_id(folder_name: str) -> str:
    """'Emma Tiffany Umwari' / 'Hassan_Adelani_Luqman' -> 'emma' / 'hassan'"""
    return re.split(r"[\s_]+", folder_name.strip())[0].lower()


def phrase_id(file_stem: str) -> str:
    s = file_stem.lower()
    if "confirm" in s:
        return "confirm_transaction"
    if "approve" in s or "yes" in s:
        return "yes_approve"
    raise ValueError(f"Cannot tell which phrase '{file_stem}' is - rename it to contain 'approve' or 'confirm'")


def to_wav(src: Path, dst: Path):
    """Decode any audio/video container to 16 kHz mono 16-bit WAV."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)]
    subprocess.run(cmd, check=True)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    takes: dict[tuple[str, str], int] = {}
    for member_dir in sorted(p for p in ORIGINAL_DIR.iterdir() if p.is_dir()):
        member = member_id(member_dir.name)
        for src in sorted(p for p in member_dir.iterdir() if p.suffix.lower() in MEDIA_EXTS):
            phrase = phrase_id(src.stem)
            takes[(member, phrase)] = takes.get((member, phrase), 0) + 1
            dst = RAW_DIR / f"{member}_{phrase}_{takes[(member, phrase)]}.wav"
            to_wav(src, dst)
            print(f"{member_dir.name}/{src.name:28s} -> {dst.relative_to(ROOT)}")
    print(f"\nConverted {sum(takes.values())} recordings for {len({m for m, _ in takes})} members.")


if __name__ == "__main__":
    main()
