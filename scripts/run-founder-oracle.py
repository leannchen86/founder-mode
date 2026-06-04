#!/usr/bin/env python3
"""Retrieve visual neighbors and aggregate text archetypes into a Founder Mode oracle readout."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
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
from PIL import Image, ImageOps


DEFAULT_IMAGE_EMBEDDINGS = "data/embeddings/ViT-L-14-quickgelu_metaclip_fullcc_image_embeddings.npz"
DEFAULT_LINKS = (
    "data/archetypes/"
    "ViT-L-14-quickgelu_metaclip_fullcc__sentence-transformers-all-MiniLM-L6-v2_mean_image_text_links.jsonl"
)
DEFAULT_CLUSTERS = "data/archetypes/sentence-transformers-all-MiniLM-L6-v2_mean_clusters.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-embeddings", default=DEFAULT_IMAGE_EMBEDDINGS)
    parser.add_argument("--links", default=DEFAULT_LINKS)
    parser.add_argument("--clusters", default=DEFAULT_CLUSTERS)
    parser.add_argument("--out-dir", default="data/oracle")
    parser.add_argument("--image", default="", help="Optional query image path. Should already be face-ish/croppable.")
    parser.add_argument("--row-index", type=int, default=-1, help="Use an existing embedding row as the query.")
    parser.add_argument("--name", default="", help="Use the first existing embedding row whose name contains this text.")
    parser.add_argument("--model", default="ViT-L-14-quickgelu")
    parser.add_argument("--pretrained", default="metaclip_fullcc")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--show-neighbors", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.06)
    parser.add_argument("--min-score", type=int, default=58)
    parser.add_argument("--min-top-similarity", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return value[:100] or "oracle"


def normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def load_image_embeddings(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "X": normalize_rows(np.asarray(data["embeddings"], dtype=np.float32)),
        "labels": np.asarray(data["labels"], dtype=np.int64),
        "label_names": [str(value) for value in data["label_names"].tolist()],
        "names": [str(value) for value in data["names"].tolist()],
        "source_keys": [str(value) for value in data["source_keys"].tolist()],
        "paths": [str(value) for value in data["paths"].tolist()] if "paths" in data.files else [],
    }


def encode_image(path: Path, model_name: str, pretrained: str, device: str) -> np.ndarray:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    batch = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.cpu().float().numpy()[0]


def find_row_index(args: argparse.Namespace, names: list[str]) -> int:
    if args.row_index >= 0:
        if args.row_index >= len(names):
            raise SystemExit(f"--row-index {args.row_index} is outside embedding rows 0..{len(names) - 1}.")
        return args.row_index
    if args.name:
        query = args.name.strip().lower()
        for index, name in enumerate(names):
            if query in name.lower():
                return index
        raise SystemExit(f"No embedding row name contains {args.name!r}.")
    return -1


def weighted_scores(similarities: np.ndarray, temperature: float) -> np.ndarray:
    if len(similarities) == 0:
        return similarities
    temp = max(temperature, 1e-6)
    shifted = (similarities - float(similarities.max())) / temp
    weights = np.exp(np.clip(shifted, -50, 50))
    return weights / max(float(weights.sum()), 1e-12)


def top_neighbor_indices(X: np.ndarray, query: np.ndarray, top_k: int, exclude_index: int) -> tuple[np.ndarray, np.ndarray]:
    sims = X @ query
    if exclude_index >= 0:
        sims[exclude_index] = -np.inf
    k = min(max(1, top_k), X.shape[0] - (1 if exclude_index >= 0 else 0))
    nearest = np.argpartition(-sims, kth=np.arange(k))[:k]
    nearest = nearest[np.argsort(-sims[nearest])]
    return nearest.astype(int), sims[nearest].astype(float)


def add_weight(bucket: defaultdict[str, float], key: str, weight: float) -> None:
    if key:
        bucket[key] += float(weight)


def sorted_weights(bucket: dict[str, float], limit: int = 10) -> list[dict[str, Any]]:
    total = sum(bucket.values()) or 1.0
    return [
        {"value": key, "weight": round(value / total, 4)}
        for key, value in sorted(bucket.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def aura_mix_percentages(anchor_weights: dict[str, float]) -> list[dict[str, Any]]:
    labels = [
        ("founders", "Founder"),
        ("vcs", "VC"),
        ("faang_engineers", "Big-Tech"),
    ]
    total = sum(anchor_weights.values()) or 1.0
    raw = [(key, label, 100 * anchor_weights.get(key, 0.0) / total) for key, label in labels]
    rounded = [(key, label, int(round(value))) for key, label, value in raw]
    delta = 100 - sum(value for _key, _label, value in rounded)
    if rounded and delta:
        largest_index = max(range(len(raw)), key=lambda index: raw[index][2])
        key, label, value = rounded[largest_index]
        rounded[largest_index] = (key, label, value + delta)
    return [{"key": key, "label": label, "percent": max(0, value)} for key, label, value in rounded]


def aura_mix_line(aura_mix: list[dict[str, Any]]) -> str:
    display = {"Founder": "founder", "VC": "VC", "Big-Tech": "big-tech"}
    return " / ".join(f"{item['percent']}% {display.get(item['label'], item['label'].lower())}" for item in aura_mix)


def tag_ratios_from_link(link: dict[str, Any]) -> dict[str, float]:
    tags = link.get("clusterCareerTags") or []
    max_count = max([int(item.get("count", 0)) for item in tags] or [1])
    return {str(item.get("value", "")): min(1.0, int(item.get("count", 0)) / max_count) for item in tags}


def compute_founder_mode(
    anchor_weights: dict[str, float],
    tag_weights: dict[str, float],
    mean_similarity: float,
    top_cluster_weight: float,
) -> int:
    total_anchor = sum(anchor_weights.values()) or 1.0
    founder = anchor_weights.get("founders", 0.0) / total_anchor
    vc = anchor_weights.get("vcs", 0.0) / total_anchor
    big_tech = anchor_weights.get("faang_engineers", 0.0) / total_anchor
    total_tags = sum(tag_weights.values()) or 1.0
    tag_signal = (
        0.35 * tag_weights.get("founder", 0.0)
        + 0.22 * tag_weights.get("founder_investor_hybrid", 0.0)
        + 0.18 * tag_weights.get("ex_big_tech_infra_founder", 0.0)
        + 0.16 * tag_weights.get("ai", 0.0)
        + 0.14 * tag_weights.get("infrastructure", 0.0)
        + 0.12 * tag_weights.get("operator", 0.0)
        + 0.10 * tag_weights.get("finance", 0.0)
    ) / total_tags
    similarity_bonus = max(0.0, min(1.0, (mean_similarity - 0.55) / 0.25))
    score = (
        14
        + 58 * founder
        + 18 * vc
        + 5 * big_tech
        + 125 * tag_signal
        + 10 * similarity_bonus
        + 10 * top_cluster_weight
    )
    return int(max(1, min(99, round(score))))


def archetype_short(name: str) -> str:
    return name.split(":", 1)[1].strip() if ":" in name else name


def fortune_for(score: int, aura_mix: list[dict[str, Any]], top_clusters: list[dict[str, Any]], top_tags: list[dict[str, Any]]) -> str:
    primary = archetype_short(top_clusters[0]["name"]) if top_clusters else "Bay Area ambiguity"
    secondary = archetype_short(top_clusters[1]["name"]) if len(top_clusters) > 1 else "network-adjacent chaos"
    tag_words = [item["value"].replace("_", " ") for item in top_tags[:4]]
    if score >= 82:
        opening = "The oracle sees a term sheet trying to happen."
    elif score >= 65:
        opening = "The oracle detects credible founder weather."
    elif score >= 45:
        opening = "The oracle sees a person one coffee away from a pitch deck."
    else:
        opening = "The oracle sees builder energy with only mild fundraising contamination."
    tags_text = ", ".join(tag_words) if tag_words else "ambiguous Bay Area signal"
    mix_text = aura_mix_line(aura_mix)
    return (
        f"{opening} Aura mix: {mix_text}. Primary aura: {primary}. Secondary aura: {secondary}. "
        f"Detected mythology: {tags_text}. Likely fake backstory: you have either said "
        f"\"platform\" unprompted, survived a strategy offsite, or described something normal as agentic."
    )


def aggregate_oracle(
    neighbor_indices: np.ndarray,
    similarities: np.ndarray,
    links: list[dict[str, Any]],
    clusters: dict[str, dict[str, Any]],
    show_neighbors: int,
    temperature: float,
    min_score: int,
    min_top_similarity: float,
) -> dict[str, Any]:
    weights = weighted_scores(similarities, temperature)
    label_weights: defaultdict[str, float] = defaultdict(float)
    anchor_weights: defaultdict[str, float] = defaultdict(float)
    cluster_weights: defaultdict[str, float] = defaultdict(float)
    tag_weights: defaultdict[str, float] = defaultdict(float)

    neighbors = []
    for rank, (index, similarity, weight) in enumerate(zip(neighbor_indices, similarities, weights), start=1):
        link = links[int(index)]
        add_weight(label_weights, link.get("imageLabel", ""), float(weight))
        add_weight(anchor_weights, link.get("clusterAnchorLabel", ""), float(weight))
        add_weight(cluster_weights, link.get("clusterId", ""), float(weight))
        for tag, ratio in tag_ratios_from_link(link).items():
            add_weight(tag_weights, tag, float(weight) * ratio)
        if rank <= show_neighbors:
            neighbors.append(
                {
                    "rank": rank,
                    "rowIndex": int(index),
                    "name": link.get("name", ""),
                    "similarity": round(float(similarity), 4),
                    "imageLabel": link.get("imageLabel", ""),
                    "clusterId": link.get("clusterId", ""),
                    "clusterName": link.get("clusterName", ""),
                    "imagePath": link.get("imagePath", ""),
                }
            )

    top_clusters = []
    for cluster_id, score in sorted(cluster_weights.items(), key=lambda item: (-item[1], item[0]))[:8]:
        cluster = clusters.get(cluster_id, {})
        top_clusters.append(
            {
                "clusterId": cluster_id,
                "name": cluster.get("name", cluster_id),
                "anchorLabel": cluster.get("anchorLabel", ""),
                "weight": round(score / max(float(sum(cluster_weights.values())), 1e-12), 4),
                "description": cluster.get("description", ""),
            }
        )

    top_tags = sorted_weights(tag_weights, limit=12)
    top_cluster_weight = max(cluster_weights.values()) / max(float(sum(cluster_weights.values())), 1e-12)
    top_similarity = float(similarities[0]) if len(similarities) else 0.0
    mean_top_10_similarity = float(np.mean(similarities[: min(10, len(similarities))])) if len(similarities) else 0.0
    founder_mode = compute_founder_mode(anchor_weights, tag_weights, mean_top_10_similarity, top_cluster_weight)
    aura_mix = aura_mix_percentages(anchor_weights)
    no_archetype_detected = founder_mode < min_score or top_similarity < min_top_similarity
    confidence = {
        "topSimilarity": round(top_similarity, 4),
        "meanTop10Similarity": round(mean_top_10_similarity, 4),
        "topClusterWeight": round(top_cluster_weight, 4),
        "minScore": min_score,
        "minTopSimilarity": min_top_similarity,
    }
    abstain_reasons = []
    if founder_mode < min_score:
        abstain_reasons.append("score_below_threshold")
    if top_similarity < min_top_similarity:
        abstain_reasons.append("top_similarity_below_threshold")

    if no_archetype_detected:
        return {
            "founderModePercent": founder_mode,
            "noArchetypeDetected": True,
            "verdict": "No archetype detected",
            "auraMix": aura_mix,
            "auraMixLine": aura_mix_line(aura_mix),
            "primaryArchetype": {
                "name": "No archetype detected",
                "weight": 1.0,
                "description": "Signal fell below the oracle threshold.",
            },
            "secondaryArchetypes": [],
            "anchorWeights": sorted_weights(anchor_weights, limit=5),
            "imageLabelWeights": sorted_weights(label_weights, limit=5),
            "careerSignalWeights": top_tags,
            "confidence": confidence,
            "abstainReasons": abstain_reasons,
            "fortune": "No archetype detected.",
            "neighbors": neighbors,
        }

    return {
        "founderModePercent": founder_mode,
        "noArchetypeDetected": False,
        "verdict": "Bay Area archetype detected",
        "auraMix": aura_mix,
        "auraMixLine": aura_mix_line(aura_mix),
        "primaryArchetype": top_clusters[0] if top_clusters else {},
        "secondaryArchetypes": top_clusters[1:5],
        "anchorWeights": sorted_weights(anchor_weights, limit=5),
        "imageLabelWeights": sorted_weights(label_weights, limit=5),
        "careerSignalWeights": top_tags,
        "confidence": confidence,
        "abstainReasons": [],
        "fortune": fortune_for(founder_mode, aura_mix, top_clusters, top_tags),
        "neighbors": neighbors,
    }


def main() -> None:
    args = parse_args()
    image_embedding_path = Path(args.image_embeddings)
    links_path = Path(args.links)
    clusters_path = Path(args.clusters)
    out_dir = Path(args.out_dir)
    payload = load_image_embeddings(image_embedding_path)
    links = read_jsonl(links_path)
    clusters_payload = json.loads(clusters_path.read_text(encoding="utf-8"))
    clusters = {cluster["clusterId"]: cluster for cluster in clusters_payload["clusters"]}

    if payload["X"].shape[0] != len(links):
        raise SystemExit(f"Embedding rows ({payload['X'].shape[0]}) do not match link rows ({len(links)}).")
    if args.image and (args.row_index >= 0 or args.name):
        raise SystemExit("Use either --image or --row-index/--name, not both.")

    query_row_index = find_row_index(args, payload["names"])
    device = choose_device(args.device)
    query_label = "external_image"
    query_meta: dict[str, Any] = {}

    print(
        json.dumps(
            {
                "imageEmbeddings": str(image_embedding_path),
                "links": str(links_path),
                "clusters": str(clusters_path),
                "rows": int(payload["X"].shape[0]),
                "queryImage": args.image,
                "queryRowIndex": query_row_index,
                "topK": args.top_k,
                "minScore": args.min_score,
                "minTopSimilarity": args.min_top_similarity,
                "device": device,
                "dryRun": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    if args.image:
        query_vector = encode_image(Path(args.image), args.model, args.pretrained, device)
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        query_label = Path(args.image).stem
        query_meta = {"type": "image", "path": args.image}
        exclude_index = -1
    elif query_row_index >= 0:
        query_vector = payload["X"][query_row_index]
        query_label = f"row_{query_row_index}_{payload['names'][query_row_index]}"
        query_meta = {"type": "existing_row", "rowIndex": query_row_index, **links[query_row_index]}
        exclude_index = query_row_index
    else:
        raise SystemExit("Provide --image, --row-index, or --name.")

    neighbor_indices, similarities = top_neighbor_indices(payload["X"], query_vector, args.top_k, exclude_index)
    oracle = aggregate_oracle(
        neighbor_indices,
        similarities,
        links,
        clusters,
        args.show_neighbors,
        args.temperature,
        args.min_score,
        args.min_top_similarity,
    )
    result = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": query_meta,
        "imageEmbeddings": str(image_embedding_path),
        "links": str(links_path),
        "clusters": str(clusters_path),
        "topK": args.top_k,
        "temperature": args.temperature,
        "minScore": args.min_score,
        "minTopSimilarity": args.min_top_similarity,
        **oracle,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{safe_slug(query_label)}_founder_oracle.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "wrote": str(output_path),
                "founderModePercent": result["founderModePercent"],
                "verdict": result["verdict"],
                "primaryArchetype": result["primaryArchetype"].get("name", ""),
                "fortune": result["fortune"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
