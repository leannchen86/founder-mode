#!/usr/bin/env python3
"""Extract frozen DINO/DINOv3 image embeddings from cleaned face crops."""

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
import timm
import torch
from PIL import Image, ImageOps
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


class CleanFacePilDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        record = self.records[index]
        image = Image.open(record["path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image, index


class CleanFaceTensorDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], transform: Any):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = Image.open(record["path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), index


def collate_images(batch: list[tuple[Image.Image, int]]) -> tuple[list[Image.Image], list[int]]:
    images, indices = zip(*batch)
    return list(images), list(indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", default="data/images/clean_manifest.jsonl")
    parser.add_argument("--out-dir", default="data/embeddings")
    parser.add_argument("--model", default="vit_base_patch16_dinov3")
    parser.add_argument("--backend", default="auto", choices=["auto", "timm", "transformers"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--feature",
        default="auto",
        choices=["auto", "pooler", "cls", "mean-patch", "timm-forward"],
        help="Which model output to save. auto uses CLS for timm ViTs and pooler_output for Transformers when available.",
    )
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


def choose_backend(requested: str, model: str) -> str:
    if requested != "auto":
        return requested
    if model.startswith("timm/") or model in timm.list_models("*"):
        return "timm"
    return "transformers"


def normalize_timm_model_name(model: str) -> str:
    if model.startswith("timm/"):
        model = model.split("/", 1)[1]
    if model.endswith(".lvd1689m"):
        model = model.removesuffix(".lvd1689m")
    return model


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


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def tensor_to_device(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def select_features(outputs: Any, feature: str, num_register_tokens: int) -> torch.Tensor:
    if feature == "auto":
        feature = "pooler" if getattr(outputs, "pooler_output", None) is not None else "cls"

    if feature == "pooler":
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            raise RuntimeError("Model did not return pooler_output. Use --feature cls or --feature mean-patch.")
        return pooled

    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("Model did not return last_hidden_state.")

    if feature == "cls":
        return hidden[:, 0, :]
    if feature == "mean-patch":
        patch_start = 1 + max(0, num_register_tokens)
        return hidden[:, patch_start:, :].mean(dim=1)
    raise RuntimeError(f"Unsupported feature output: {feature}")


def select_timm_features(model: torch.nn.Module, batch: torch.Tensor, feature: str) -> tuple[torch.Tensor, str]:
    if feature == "timm-forward":
        return model(batch), "timm-forward"

    hidden = model.forward_features(batch)
    if isinstance(hidden, dict):
        for key in ("x", "last_hidden_state", "features"):
            if key in hidden:
                hidden = hidden[key]
                break

    if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
        return hidden, "timm-forward"
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise RuntimeError(f"Unsupported timm feature output shape: {type(hidden)!r} {getattr(hidden, 'shape', None)!r}")

    selected = "cls" if feature == "auto" else feature
    if selected == "pooler":
        return model.forward_head(hidden, pre_logits=True), "pooler"
    if selected == "cls":
        return hidden[:, 0, :], "cls"
    if selected == "mean-patch":
        patch_start = int(getattr(model, "num_prefix_tokens", 1) or 1)
        return hidden[:, patch_start:, :].mean(dim=1), "mean-patch"

    raise RuntimeError(f"Unsupported timm feature output: {feature}")


def extract_transformers_embeddings(
    records: list[dict[str, Any]],
    model_name: str,
    device: str,
    feature: str,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    dataset = CleanFacePilDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_images,
    )
    embeddings: list[np.ndarray] = []
    selected_feature: str | None = None
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0) or 0)

    with torch.inference_mode():
        for images, _indices in loader:
            inputs = processor(images=images, return_tensors="pt")
            inputs = tensor_to_device(inputs, device)
            outputs = model(**inputs)
            feats = select_features(outputs, feature, num_register_tokens)
            if selected_feature is None:
                if feature == "auto":
                    selected_feature = "pooler" if getattr(outputs, "pooler_output", None) is not None else "cls"
                else:
                    selected_feature = feature
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            embeddings.append(feats.cpu().float().numpy())

    return (
        np.concatenate(embeddings, axis=0).astype(np.float32),
        selected_feature or feature,
        {"numRegisterTokens": num_register_tokens},
    )


def extract_timm_embeddings(
    records: list[dict[str, Any]],
    model_name: str,
    device: str,
    feature: str,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    timm_model_name = normalize_timm_model_name(model_name)
    model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
    model.to(device)
    model.eval()

    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    dataset = CleanFaceTensorDataset(records, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    embeddings: list[np.ndarray] = []
    selected_feature: str | None = None
    with torch.inference_mode():
        for batch, _indices in loader:
            batch = batch.to(device)
            feats, selected = select_timm_features(model, batch, feature)
            if selected_feature is None:
                selected_feature = selected
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            embeddings.append(feats.cpu().float().numpy())

    return (
        np.concatenate(embeddings, axis=0).astype(np.float32),
        selected_feature or feature,
        {"timmModelName": timm_model_name, "timmDataConfig": data_config},
    )


def main() -> None:
    args = parse_args()
    clean_manifest = Path(args.clean_manifest)
    records = read_clean_records(clean_manifest, args.limit)
    device = choose_device(args.device)
    backend = choose_backend(args.backend, args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "cleanManifest": args.clean_manifest,
                "outDir": str(out_dir),
                "records": len(records),
                "model": args.model,
                "backend": backend,
                "batchSize": args.batch_size,
                "feature": args.feature,
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
    if backend == "timm":
        X, selected_feature, extra_config = extract_timm_embeddings(
            records,
            args.model,
            device,
            args.feature,
            args.batch_size,
            args.num_workers,
        )
    else:
        X, selected_feature, extra_config = extract_transformers_embeddings(
            records,
            args.model,
            device,
            args.feature,
            args.batch_size,
            args.num_workers,
        )

    label_names, y = labels(records)
    paths = np.array([record["path"] for record in records], dtype=object)
    source_keys = np.array([record.get("sourceKey", "") for record in records], dtype=object)
    names = np.array([record.get("name", "") for record in records], dtype=object)

    slug = safe_slug(args.model.replace("/", "-"))
    feature_slug = safe_slug(selected_feature)
    npz_path = out_dir / f"{slug}_{feature_slug}_image_embeddings.npz"
    manifest_path = out_dir / f"{slug}_{feature_slug}_image_embeddings_manifest.jsonl"
    config_path = out_dir / f"{slug}_{feature_slug}_image_embeddings_config.json"

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
        "backend": backend,
        "device": device,
        "feature": selected_feature,
        "embeddingDim": int(X.shape[1]),
        "records": int(X.shape[0]),
        "labelNames": label_names,
        "sourceManifest": str(clean_manifest.resolve()),
        "elapsedSeconds": round(time.time() - started, 3),
    }
    config.update(extra_config)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(npz_path), "shape": list(X.shape), "config": str(config_path)}, indent=2))


if __name__ == "__main__":
    main()
