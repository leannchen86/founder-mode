#!/usr/bin/env python3
"""Evaluate image embeddings and create visual QA sheets."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default="data/embeddings/ViT-L-14_openai_image_embeddings.npz")
    parser.add_argument("--out-dir", default="data/eval")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-sheet-per-class", type=int, default=18)
    parser.add_argument("--retrieval-queries-per-class", type=int, default=4)
    parser.add_argument("--retrieval-neighbors", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def slug_from_path(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix("_image_embeddings")


def load_embeddings(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    required = {"embeddings", "labels", "label_names", "paths"}
    missing = sorted(required - set(data.files))
    if missing:
        raise SystemExit(f"{path} is missing required arrays: {', '.join(missing)}")

    names = data["names"] if "names" in data.files else np.array([""] * len(data["labels"]), dtype=object)
    source_keys = (
        data["source_keys"] if "source_keys" in data.files else np.array([""] * len(data["labels"]), dtype=object)
    )
    return {
        "X": np.asarray(data["embeddings"], dtype=np.float32),
        "y": np.asarray(data["labels"], dtype=np.int64),
        "label_names": [str(label) for label in data["label_names"].tolist()],
        "paths": [str(path) for path in data["paths"].tolist()],
        "names": [str(name) for name in names.tolist()],
        "source_keys": [str(key) for key in source_keys.tolist()],
    }


def choose_folds(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    max_folds = int(counts.min())
    folds = min(requested, max_folds)
    if folds < 2:
        raise SystemExit("Need at least two examples per class for cross-validation.")
    return folds


def evaluate_estimator(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    estimator: Any,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y)

    fold_scores = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        estimator.fit(X[train_index], y[train_index])
        fold_pred = estimator.predict(X[test_index])
        y_pred[test_index] = fold_pred
        fold_scores.append(
            {
                "fold": fold_index,
                "n": int(len(test_index)),
                "accuracy": float(accuracy_score(y[test_index], fold_pred)),
                "balancedAccuracy": float(balanced_accuracy_score(y[test_index], fold_pred)),
                "macroF1": float(f1_score(y[test_index], fold_pred, average="macro")),
            }
        )

    return {
        "folds": fold_scores,
        "accuracy": float(accuracy_score(y, y_pred)),
        "balancedAccuracy": float(balanced_accuracy_score(y, y_pred)),
        "macroF1": float(f1_score(y, y_pred, average="macro")),
        "weightedF1": float(f1_score(y, y_pred, average="weighted")),
        "confusionMatrix": confusion_matrix(y, y_pred).astype(int).tolist(),
        "classificationReport": classification_report(
            y,
            y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def nearest_label_metrics(X: np.ndarray, y: np.ndarray, top_k: int) -> dict[str, Any]:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized = X / np.maximum(norms, 1e-12)
    sims = normalized @ normalized.T
    np.fill_diagonal(sims, -np.inf)

    k = min(top_k, X.shape[0] - 1)
    nearest = np.argpartition(-sims, kth=np.arange(k), axis=1)[:, :k]
    nearest = nearest[np.arange(nearest.shape[0])[:, None], np.argsort(-sims[np.arange(nearest.shape[0])[:, None], nearest])]
    same = y[nearest] == y[:, None]

    return {
        "top1SameLabel": float(same[:, 0].mean()),
        f"top{k}SameLabelMean": float(same.mean()),
        "topK": int(k),
    }


def pil_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def fit_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    ellipsis = "..."
    while text and draw.textlength(text + ellipsis, font=font) > max_width:
        text = text[:-1]
    return text + ellipsis if text else ellipsis


def image_cell(
    path: str,
    title: str,
    subtitle: str,
    size: int,
    footer: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    border: str | None = None,
) -> Image.Image:
    cell = Image.new("RGB", (size, size + footer), "white")
    try:
        image = Image.open(path).convert("RGB")
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        cell.paste(image, (0, 0))
    except Exception:
        placeholder = Image.new("RGB", (size, size), "#eeeeee")
        cell.paste(placeholder, (0, 0))

    draw = ImageDraw.Draw(cell)
    if border:
        draw.rectangle((0, 0, size - 1, size - 1), outline=border, width=5)

    draw.rectangle((0, size, size, size + footer), fill="white")
    draw.text((6, size + 5), fit_text(title, draw, font, size - 12), fill="#111111", font=font)
    draw.text((6, size + 24), fit_text(subtitle, draw, small_font, size - 12), fill="#555555", font=small_font)
    return cell


def make_contact_sheet(
    paths: list[str],
    names: list[str],
    y: np.ndarray,
    label_names: list[str],
    out_path: Path,
    per_class: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    cell_size = 128
    footer = 44
    gap = 8
    label_width = 150
    font = pil_font(13)
    small_font = pil_font(11)

    rows: list[tuple[str, list[int]]] = []
    for label_index, label in enumerate(label_names):
        indices = np.where(y == label_index)[0].tolist()
        rng.shuffle(indices)
        rows.append((label, indices[:per_class]))

    cols = max(len(indices) for _, indices in rows)
    width = label_width + cols * cell_size + (cols + 1) * gap
    height = gap + len(rows) * (cell_size + footer + gap)
    sheet = Image.new("RGB", (width, height), "#f4f4f4")
    draw = ImageDraw.Draw(sheet)

    for row_index, (label, indices) in enumerate(rows):
        top = gap + row_index * (cell_size + footer + gap)
        draw.text((16, top + 12), label, fill="#111111", font=pil_font(18))
        draw.text((16, top + 38), f"sample {len(indices)}", fill="#555555", font=font)
        for col_index, sample_index in enumerate(indices):
            left = label_width + gap + col_index * (cell_size + gap)
            cell = image_cell(
                paths[sample_index],
                names[sample_index] or Path(paths[sample_index]).stem,
                label,
                cell_size,
                footer,
                font,
                small_font,
            )
            sheet.paste(cell, (left, top))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def nearest_indices(X: np.ndarray, query_index: int, count: int) -> list[int]:
    query = X[query_index]
    norms = np.linalg.norm(X, axis=1) * max(np.linalg.norm(query), 1e-12)
    sims = (X @ query) / np.maximum(norms, 1e-12)
    sims[query_index] = -np.inf
    k = min(count, len(sims) - 1)
    nearest = np.argpartition(-sims, kth=np.arange(k))[:k]
    nearest = nearest[np.argsort(-sims[nearest])]
    return nearest.astype(int).tolist()


def make_retrieval_sheet(
    X: np.ndarray,
    paths: list[str],
    names: list[str],
    y: np.ndarray,
    label_names: list[str],
    out_path: Path,
    queries_per_class: int,
    neighbors: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    cell_size = 128
    footer = 44
    gap = 8
    label_width = 180
    font = pil_font(13)
    small_font = pil_font(11)

    query_indices: list[int] = []
    for label_index in range(len(label_names)):
        indices = np.where(y == label_index)[0].tolist()
        rng.shuffle(indices)
        query_indices.extend(indices[:queries_per_class])

    cols = neighbors + 1
    width = label_width + cols * cell_size + (cols + 1) * gap
    height = gap + len(query_indices) * (cell_size + footer + gap)
    sheet = Image.new("RGB", (width, height), "#f4f4f4")
    draw = ImageDraw.Draw(sheet)

    for row_index, query_index in enumerate(query_indices):
        top = gap + row_index * (cell_size + footer + gap)
        label = label_names[int(y[query_index])]
        draw.text((16, top + 10), "query", fill="#111111", font=font)
        draw.text((16, top + 30), fit_text(names[query_index] or Path(paths[query_index]).stem, draw, font, 155), fill="#111111", font=font)
        draw.text((16, top + 50), label, fill="#555555", font=small_font)

        row_indices = [query_index] + nearest_indices(X, query_index, neighbors)
        for col_index, sample_index in enumerate(row_indices):
            left = label_width + gap + col_index * (cell_size + gap)
            title = "QUERY" if col_index == 0 else names[sample_index] or Path(paths[sample_index]).stem
            sample_label = label_names[int(y[sample_index])]
            border = "#111111" if col_index == 0 else ("#239a53" if y[sample_index] == y[query_index] else "#c73d32")
            cell = image_cell(
                paths[sample_index],
                title,
                sample_label,
                cell_size,
                footer,
                font,
                small_font,
                border=border,
            )
            sheet.paste(cell, (left, top))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def save_confusion_matrix(matrix: list[list[int]], label_names: list[str], out_path: Path, title: str) -> None:
    arr = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(label_names)), labels=label_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(label_names)), labels=label_names)

    threshold = arr.max() / 2 if arr.size else 0
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            ax.text(
                col,
                row,
                str(int(arr[row, col])),
                ha="center",
                va="center",
                color="white" if arr[row, col] > threshold else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    embeddings_path = Path(args.embeddings)
    out_dir = Path(args.out_dir)
    payload = load_embeddings(embeddings_path)
    X: np.ndarray = payload["X"]
    y: np.ndarray = payload["y"]
    label_names: list[str] = payload["label_names"]
    paths: list[str] = payload["paths"]
    names: list[str] = payload["names"]

    folds = choose_folds(y, args.folds)
    slug = slug_from_path(embeddings_path)

    label_counts = {label_names[int(label)]: int(count) for label, count in zip(*np.unique(y, return_counts=True))}
    plan = {
        "embeddings": str(embeddings_path),
        "shape": list(X.shape),
        "labels": label_counts,
        "folds": folds,
        "outDir": str(out_dir),
        "dryRun": args.dry_run,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    linear_probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=args.max_iter, class_weight="balanced", random_state=args.seed),
    )
    knn = KNeighborsClassifier(n_neighbors=11, metric="cosine", weights="distance", algorithm="brute")

    linear_results = evaluate_estimator(X, y, label_names, linear_probe, folds, args.seed)
    knn_results = evaluate_estimator(X, y, label_names, knn, folds, args.seed)
    retrieval_results = nearest_label_metrics(X, y, top_k=args.retrieval_neighbors)

    contact_sheet_path = out_dir / f"{slug}_contact_sheet.jpg"
    retrieval_sheet_path = out_dir / f"{slug}_nearest_neighbors.jpg"
    linear_cm_path = out_dir / f"{slug}_linear_probe_confusion.png"
    knn_cm_path = out_dir / f"{slug}_knn_confusion.png"
    results_path = out_dir / f"{slug}_eval.json"

    make_contact_sheet(
        paths,
        names,
        y,
        label_names,
        contact_sheet_path,
        per_class=args.contact_sheet_per_class,
        seed=args.seed,
    )
    make_retrieval_sheet(
        X,
        paths,
        names,
        y,
        label_names,
        retrieval_sheet_path,
        queries_per_class=args.retrieval_queries_per_class,
        neighbors=args.retrieval_neighbors,
        seed=args.seed,
    )
    save_confusion_matrix(
        linear_results["confusionMatrix"],
        label_names,
        linear_cm_path,
        "Linear probe confusion matrix",
    )
    save_confusion_matrix(knn_results["confusionMatrix"], label_names, knn_cm_path, "kNN confusion matrix")

    result = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embeddingPath": str(embeddings_path),
        "shape": list(X.shape),
        "labelNames": label_names,
        "labelCounts": label_counts,
        "folds": folds,
        "seed": args.seed,
        "linearProbe": linear_results,
        "knnCosine": knn_results,
        "nearestNeighborPurity": retrieval_results,
        "artifacts": {
            "contactSheet": str(contact_sheet_path),
            "nearestNeighbors": str(retrieval_sheet_path),
            "linearConfusionMatrix": str(linear_cm_path),
            "knnConfusionMatrix": str(knn_cm_path),
        },
        "elapsedSeconds": round(time.time() - started, 3),
    }
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "wrote": str(results_path),
                "linearProbe": {
                    "accuracy": result["linearProbe"]["accuracy"],
                    "balancedAccuracy": result["linearProbe"]["balancedAccuracy"],
                    "macroF1": result["linearProbe"]["macroF1"],
                },
                "knnCosine": {
                    "accuracy": result["knnCosine"]["accuracy"],
                    "balancedAccuracy": result["knnCosine"]["balancedAccuracy"],
                    "macroF1": result["knnCosine"]["macroF1"],
                },
                "nearestNeighborPurity": result["nearestNeighborPurity"],
                "artifacts": result["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
