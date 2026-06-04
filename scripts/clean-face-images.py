#!/usr/bin/env python3
"""Clean downloaded profile images into face crops using batch_face RetinaFace."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import cv2
import numpy as np
from PIL import Image, ImageOps
from batch_face import RetinaFace


@dataclass
class DetectedFace:
    crop: Image.Image
    bbox: tuple[int, int, int, int]
    score: float
    area: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/images/raw_manifest.jsonl")
    parser.add_argument("--out-dir", default="data/images/clean")
    parser.add_argument("--clean-manifest", default="data/images/clean_manifest.jsonl")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--gpu-id", type=int, default=-1, help="-1 for CPU, 0 for GPU/CUDA-style devices if available.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_face(face: Any) -> tuple[tuple[float, float, float, float], float] | None:
    try:
        if isinstance(face, (list, tuple)) and len(face) >= 3:
            score = float(face[2])
            bbox = face[0]
            if hasattr(bbox, "tolist"):
                bbox = bbox.tolist()
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = map(float, bbox[:4])
                return (x1, y1, x2, y2), score
        if isinstance(face, (list, tuple)) and len(face) >= 5:
            x1, y1, x2, y2, score = map(float, face[:5])
            return (x1, y1, x2, y2), score
    except Exception:
        return None
    return None


def square_crop_xyxy(bbox: tuple[float, float, float, float], margin: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(width, height) * (1.0 + margin)
    half = side / 2.0
    return (
        int(round(center_x - half)),
        int(round(center_y - half)),
        int(round(center_x + half)),
        int(round(center_y + half)),
    )


def pad_reflect_and_crop(src_rgb: np.ndarray, crop_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = crop_xyxy
    height, width = src_rgb.shape[:2]
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    if any(value > 0 for value in (pad_left, pad_top, pad_right, pad_bottom)):
        src_rgb = cv2.copyMakeBorder(
            src_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

    px1 = max(0, min(src_rgb.shape[1] - 1, x1 + pad_left))
    py1 = max(0, min(src_rgb.shape[0] - 1, y1 + pad_top))
    px2 = max(px1 + 1, min(src_rgb.shape[1], x2 + pad_left))
    py2 = max(py1 + 1, min(src_rgb.shape[0], y2 + pad_top))
    return src_rgb[py1:py2, px1:px2]


def detect_faces(detector: RetinaFace, image: Image.Image, threshold: float, margin: float) -> list[DetectedFace]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.array(image)
    faces = detector([rgb], threshold=threshold, batch_size=1)
    faces = faces[0] if faces else []

    detected: list[DetectedFace] = []
    for face in faces:
        parsed = parse_face(face)
        if parsed is None:
            continue
        bbox_float, score = parsed
        x1, y1, x2, y2 = bbox_float
        bbox = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
        crop = Image.fromarray(pad_reflect_and_crop(rgb, square_crop_xyxy(bbox_float, margin)))
        detected.append(DetectedFace(crop=crop, bbox=bbox, score=score, area=area))
    return detected


def read_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") in {"downloaded", "skipped_existing"} and record.get("path"):
                records.append(record)
                if limit and len(records) >= limit:
                    break
    return records


def output_path(out_dir: Path, record: dict[str, Any]) -> Path:
    labels = record.get("archetypeLabels") or ["unlabeled"]
    label = str(labels[0]).replace("/", "_")
    return out_dir / label / f"{record['key']}.jpg"


def main() -> None:
    args = parse_args()
    records = read_manifest(Path(args.manifest), args.limit)
    print(
        json.dumps(
            {
                "manifest": args.manifest,
                "outDir": args.out_dir,
                "records": len(records),
                "threshold": args.threshold,
                "margin": args.margin,
                "imageSize": args.image_size,
                "gpuId": args.gpu_id,
                "dryRun": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    detector = RetinaFace(gpu_id=args.gpu_id)
    out_dir = Path(args.out_dir)
    clean_manifest = Path(args.clean_manifest)
    clean_manifest.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    with clean_manifest.open("w", encoding="utf-8") as manifest:
        for record in records:
            result = {
                "sourcePath": record["path"],
                "sourceUrl": record.get("url", ""),
                "sourceKey": record["key"],
                "name": record.get("name", ""),
                "archetypeLabels": record.get("archetypeLabels") or [],
            }
            try:
                image = Image.open(record["path"])
                faces = detect_faces(detector, image, args.threshold, args.margin)
            except Exception as error:
                status = "failed"
                result.update({"status": status, "error": str(error)})
            else:
                if not faces:
                    status = "no_face"
                    result.update({"status": status})
                else:
                    best = max(faces, key=lambda face: (face.score, face.area))
                    path = output_path(out_dir, record)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    crop = best.crop.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
                    crop.save(path, quality=94)
                    status = "cleaned"
                    result.update(
                        {
                            "status": status,
                            "path": str(path),
                            "faces": len(faces),
                            "bbox": best.bbox,
                            "score": best.score,
                            "area": best.area,
                        }
                    )

            counts[status] = counts.get(status, 0) + 1
            manifest.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps({"counts": counts}, indent=2))


if __name__ == "__main__":
    main()
