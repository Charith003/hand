"""Build SignSpeak sequences from Kaggle sign-language datasets.

This script converts image/video class folders into:
    dataset/<class>/<sample_idx>/sequence.npy   shape (30, 126)

It uses MediaPipe Hands to extract up to 2-hand landmarks per frame and creates
fixed-length sequences expected by train_lstm.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

SEQ_LENGTH = 30
FEATURES = 126
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

mp_hands = mp.solutions.hands


def extract_keypoints(results) -> np.ndarray:
    out = np.zeros(FEATURES, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return out
    for i, hand_lms in enumerate(results.multi_hand_landmarks[:2]):
        arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
        out[i * 63 : (i + 1) * 63] = arr.flatten()
    return out


def iter_class_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            yield child


def sample_frames_from_video(video_path: Path, seq_length: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return []

    idxs = np.linspace(0, len(frames) - 1, seq_length).astype(int)
    return [frames[i] for i in idxs]


def resize_or_pad(seq: list[np.ndarray], seq_length: int) -> np.ndarray:
    if len(seq) == 0:
        return np.zeros((seq_length, FEATURES), dtype=np.float32)
    if len(seq) >= seq_length:
        idxs = np.linspace(0, len(seq) - 1, seq_length).astype(int)
        out = [seq[i] for i in idxs]
    else:
        out = seq + [seq[-1]] * (seq_length - len(seq))
    return np.array(out, dtype=np.float32)


def build_from_images(
    class_dir: Path,
    out_dir: Path,
    hands: mp.solutions.hands.Hands,
    seq_length: int,
    windows_per_class: int,
    step: int,
) -> int:
    images = [p for p in sorted(class_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        return 0

    landmarks: list[np.ndarray] = []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        landmarks.append(extract_keypoints(results))

    if len(landmarks) < 2:
        return 0

    samples = 0
    cursor = 0
    while cursor < len(landmarks) and samples < windows_per_class:
        chunk = landmarks[cursor : cursor + seq_length]
        seq = resize_or_pad(chunk, seq_length)
        sample_dir = out_dir / str(samples)
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "sequence.npy", seq)
        samples += 1
        cursor += max(1, step)

    return samples


def build_from_videos(
    class_dir: Path,
    out_dir: Path,
    hands: mp.solutions.hands.Hands,
    seq_length: int,
    max_videos_per_class: int,
) -> int:
    videos = [p for p in sorted(class_dir.rglob("*")) if p.suffix.lower() in VIDEO_EXTS]
    if max_videos_per_class > 0:
        videos = videos[:max_videos_per_class]

    samples = 0
    for video in videos:
        frames = sample_frames_from_video(video, seq_length)
        if not frames:
            continue

        landmarks: list[np.ndarray] = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            landmarks.append(extract_keypoints(results))

        seq = resize_or_pad(landmarks, seq_length)
        sample_dir = out_dir / str(samples)
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "sequence.npy", seq)
        samples += 1

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-roots", nargs="+", required=True, help="One or more dataset roots")
    parser.add_argument("--output", default="dataset", help="Output dataset directory")
    parser.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    parser.add_argument("--max-classes", type=int, default=0, help="0 means all")
    parser.add_argument("--max-videos-per-class", type=int, default=100)
    parser.add_argument("--windows-per-class", type=int, default=100)
    parser.add_argument("--image-step", type=int, default=10, help="Stride for image windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    class_count = 0
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for root_str in args.dataset_roots:
            root = Path(root_str)
            if not root.exists():
                print(f"[skip] root does not exist: {root}")
                continue

            for class_dir in iter_class_dirs(root):
                class_name = class_dir.name.strip().lower().replace(" ", "_")
                class_out = out_root / class_name
                class_out.mkdir(parents=True, exist_ok=True)

                n_vid = build_from_videos(
                    class_dir,
                    class_out,
                    hands,
                    seq_length=args.seq_length,
                    max_videos_per_class=args.max_videos_per_class,
                )
                n_img = build_from_images(
                    class_dir,
                    class_out,
                    hands,
                    seq_length=args.seq_length,
                    windows_per_class=args.windows_per_class,
                    step=args.image_step,
                )
                total = n_vid + n_img
                if total > 0:
                    class_count += 1
                    print(f"[ok] {class_name}: {total} samples (videos={n_vid}, images={n_img})")

                if args.max_classes > 0 and class_count >= args.max_classes:
                    break

            if args.max_classes > 0 and class_count >= args.max_classes:
                break

    print(f"Done. Built {class_count} classes into {out_root}")


if __name__ == "__main__":
    main()
