#!/usr/bin/env python3
"""Extract frozen CLIP image embeddings from cleaned face crops."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CleanFaceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], preprocess: Any):
        self.records = records
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        return self.preprocess(image), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", default="data/images/clean_manifest.jsonl")
    parser.add_argument("--out-dir", default="data/embeddings")
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_clean_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "cleaned" or not record.get("path"):
                continue
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def labels(records: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    names = sorted({label for record in records for label in (record.get("archetypeLabels") or [])})
    label_to_index = {label: index for index, label in enumerate(names)}
    y = np.array([label_to_index[(record.get("archetypeLabels") or ["unlabeled"])[0]] for record in records], dtype=np.int64)
    return names, y


def main() -> None:
    args = parse_args()
    records = read_clean_records(Path(args.clean_manifest), args.limit)
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "cleanManifest": args.clean_manifest,
                "outDir": str(out_dir),
                "records": len(records),
                "model": args.model,
                "pretrained": args.pretrained,
                "batchSize": args.batch_size,
                "device": device,
                "dryRun": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    if not records:
        raise SystemExit("No cleaned records to embed.")

    started = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
    )
    model.eval()

    dataset = CleanFaceDataset(records, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for batch, _indices in loader:
            batch = batch.to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().float().numpy())

    X = np.concatenate(embeddings, axis=0).astype(np.float32)
    label_names, y = labels(records)
    paths = np.array([record["path"] for record in records], dtype=object)
    source_keys = np.array([record.get("sourceKey", "") for record in records], dtype=object)
    names = np.array([record.get("name", "") for record in records], dtype=object)

    slug = f"{args.model.replace('/', '-')}_{args.pretrained.replace('/', '-')}".replace(" ", "_")
    npz_path = out_dir / f"{slug}_image_embeddings.npz"
    manifest_path = out_dir / f"{slug}_image_embeddings_manifest.jsonl"
    config_path = out_dir / f"{slug}_image_embeddings_config.json"

    np.savez_compressed(
        npz_path,
        embeddings=X,
        labels=y,
        label_names=np.array(label_names, dtype=object),
        paths=paths,
        source_keys=source_keys,
        names=names,
    )
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, record in enumerate(records):
            manifest.write(
                json.dumps(
                    {
                        "rowIndex": index,
                        "embeddingPath": str(npz_path),
                        "label": label_names[int(y[index])],
                        "path": record["path"],
                        "sourceKey": record.get("sourceKey", ""),
                        "name": record.get("name", ""),
                        "archetypeLabels": record.get("archetypeLabels") or [],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    config = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "pretrained": args.pretrained,
        "device": device,
        "embeddingDim": int(X.shape[1]),
        "records": int(X.shape[0]),
        "labelNames": label_names,
        "sourceManifest": str(Path(args.clean_manifest).resolve()),
        "elapsedSeconds": round(time.time() - started, 3),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(npz_path), "shape": list(X.shape), "config": str(config_path)}, indent=2))


if __name__ == "__main__":
    main()

