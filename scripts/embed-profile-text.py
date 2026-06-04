#!/usr/bin/env python3
"""Extract local text embeddings for Diffbot profile dossiers."""

from __future__ import annotations

import argparse
import json
import os
import re
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
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/profile_dossiers_latest.jsonl")
    parser.add_argument("--out-dir", default="data/text_embeddings")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    parser.add_argument("--text-field", default="dossierText")
    parser.add_argument("--prefix", default="")
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


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def label_array(records: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    names = sorted({label for record in records for label in (record.get("archetypeLabels") or [])})
    if not names:
        names = ["unlabeled"]
    label_to_index = {label: index for index, label in enumerate(names)}
    labels = []
    for record in records:
        labels.append(label_to_index[(record.get("archetypeLabels") or ["unlabeled"])[0]])
    return names, np.asarray(labels, dtype=np.int64)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str,
    max_length: int,
    pooling: str,
) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded)
        if pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            embeddings = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return embeddings.cpu().float().numpy()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    records = read_jsonl(input_path, args.limit)
    device = choose_device(args.device)
    print(
        json.dumps(
            {
                "input": str(input_path),
                "outDir": str(out_dir),
                "records": len(records),
                "model": args.model,
                "batchSize": args.batch_size,
                "maxLength": args.max_length,
                "pooling": args.pooling,
                "device": device,
                "dryRun": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    if not records:
        raise SystemExit("No dossier records to embed.")

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    texts = [f"{args.prefix}{record.get(args.text_field, '')}" for record in records]
    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), args.batch_size):
        embeddings.append(
            embed_batch(
                model,
                tokenizer,
                texts[start : start + args.batch_size],
                device,
                args.max_length,
                args.pooling,
            )
        )

    X = np.concatenate(embeddings, axis=0).astype(np.float32)
    label_names, y = label_array(records)
    names = np.array([record.get("name", "") for record in records], dtype=object)
    source_keys = np.array([record.get("sourceKey", "") for record in records], dtype=object)
    primary_labels = np.array([record.get("primaryLabel", "") for record in records], dtype=object)

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_slug(args.model)
    npz_path = out_dir / f"{slug}_{args.pooling}_profile_text_embeddings.npz"
    manifest_path = out_dir / f"{slug}_{args.pooling}_profile_text_embeddings_manifest.jsonl"
    config_path = out_dir / f"{slug}_{args.pooling}_profile_text_embeddings_config.json"

    np.savez_compressed(
        npz_path,
        embeddings=X,
        labels=y,
        label_names=np.array(label_names, dtype=object),
        names=names,
        source_keys=source_keys,
        primary_labels=primary_labels,
    )

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, record in enumerate(records):
            text = record.get(args.text_field, "")
            manifest.write(
                json.dumps(
                    {
                        "rowIndex": index,
                        "embeddingPath": str(npz_path),
                        "name": record.get("name", ""),
                        "sourceKey": record.get("sourceKey", ""),
                        "primaryLabel": record.get("primaryLabel", ""),
                        "archetypeLabels": record.get("archetypeLabels") or [],
                        "careerTags": record.get("careerTags") or [],
                        "textPreview": text[:500],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    config = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": str(input_path.resolve()),
        "model": args.model,
        "pooling": args.pooling,
        "device": device,
        "embeddingDim": int(X.shape[1]),
        "records": int(X.shape[0]),
        "labelNames": label_names,
        "maxLength": args.max_length,
        "textField": args.text_field,
        "elapsedSeconds": round(time.time() - started, 3),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(npz_path), "shape": list(X.shape), "config": str(config_path)}, indent=2))


if __name__ == "__main__":
    main()
