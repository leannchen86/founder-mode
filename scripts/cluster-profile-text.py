#!/usr/bin/env python3
"""Cluster profile text embeddings into Bay Area archetype candidates."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossiers", default="data/processed/profile_dossiers_latest.jsonl")
    parser.add_argument("--embeddings", default="data/text_embeddings/sentence-transformers-all-MiniLM-L6-v2_mean_profile_text_embeddings.npz")
    parser.add_argument("--out-dir", default="data/archetypes")
    parser.add_argument("--clusters-per-label", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representatives", type=int, default=8)
    parser.add_argument("--top-terms", type=int, default=14)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def slug_from_path(path: Path) -> str:
    return path.stem.removesuffix("_profile_text_embeddings")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def top_counts(values: list[str], limit: int) -> list[dict[str, Any]]:
    counts = Counter(value for value in values if value)
    return [{"value": value, "count": count} for value, count in counts.most_common(limit)]


def cosine_distance_to_centroid(X: np.ndarray, indices: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    cluster_X = X[indices]
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    sims = cluster_X @ centroid
    return 1.0 - sims


EXTRA_STOP_WORDS = {
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "xx",
    "management",
    "executive",
    "member",
    "members",
    "board",
    "current",
    "past",
    "employment",
    "traces",
    "education",
    "detected",
    "career",
    "signals",
}


def terms_for_cluster(texts: list[str], top_terms: int) -> list[str]:
    if not texts:
        return []
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2 if len(texts) >= 8 else 1,
        max_df=0.85,
        max_features=4000,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    names = np.asarray(vectorizer.get_feature_names_out())
    order = np.argsort(-scores)
    terms = []
    for index in order:
        term = names[index]
        if len(term) < 3 or re.search(r"\b(d?\d{2,4}|xx)\b", term):
            continue
        if any(token in EXTRA_STOP_WORDS for token in term.split()):
            continue
        terms.append(term)
        if len(terms) >= top_terms:
            break
    return terms


def label_display(label: str) -> str:
    return {
        "founders": "Founder",
        "vcs": "VC",
        "faang_engineers": "Big-Tech Engineer",
    }.get(label, label.replace("_", " ").title())


def cluster_name(anchor: str, tag_counts: Counter[str], top_terms: list[str], size: int) -> str:
    tags = set(tag_counts)
    display = label_display(anchor)

    def ratio(tag: str) -> float:
        return tag_counts[tag] / max(1, size)

    if anchor == "faang_engineers":
        if ratio("ai") >= 0.75 and ratio("research") >= 0.65:
            return "Big-Tech Engineer: AI research engineer"
        if ratio("founder") >= 0.65 and ratio("product") >= 0.45:
            return "Big-Tech Engineer: founder-curious product/platform person"
        if ratio("product") >= 0.50 or ratio("marketplace") >= 0.45:
            return "Big-Tech Engineer: product-platform operator"
        if ratio("research") >= 0.55:
            return "Big-Tech Engineer: research-heavy systems builder"
        if ratio("infrastructure") >= 0.70:
            return "Big-Tech Engineer: infrastructure systems builder"
        return "Big-Tech Engineer: big-tech technical lane"

    if anchor == "founders":
        if ratio("health") >= 0.45:
            return "Founder: health/bio founder"
        if ratio("ai") >= 0.70 and ratio("research") >= 0.55:
            return "Founder: AI research founder"
        if ratio("finance_crypto_ai_rotation") >= 0.03:
            return "Founder: narrative-rotation founder"
        if ratio("crypto") >= 0.10 and ratio("finance") >= 0.40:
            return "Founder: finance-to-crypto operator"
        if ratio("finance") >= 0.60:
            return "Founder: finance/operator founder"
        if ratio("product") >= 0.60 or ratio("marketplace") >= 0.55:
            return "Founder: product-market operator"
        if ratio("infrastructure") >= 0.65 and ratio("big_tech") >= 0.25:
            return "Founder: ex-big-tech infrastructure founder"
        if ratio("stanford") >= 0.30 and ratio("research") >= 0.45:
            return "Founder: Stanford research circuit"
        return "Founder: startup generalist"

    if anchor == "vcs":
        if ratio("health") >= 0.45:
            return "VC: health/bio investor"
        if ratio("new_york") >= 0.45 and ratio("finance") >= 0.65:
            return "VC: New York finance to Sand Hill"
        if ratio("technical") >= 0.65 and ratio("infrastructure") >= 0.55 and ratio("ai") >= 0.40:
            return "VC: technical AI/infrastructure investor"
        if ratio("founder_investor_hybrid") >= 0.55:
            return "VC: founder-investor hybrid"
        if ratio("operator") >= 0.60 and ratio("product") >= 0.50:
            return "VC: operator-turned-investor"
        if ratio("marketplace") >= 0.55:
            return "VC: consumer/marketplace investor"
        if ratio("finance") >= 0.65:
            return "VC: finance-native investor"
        return "VC: venture generalist"

    if "stanford" in tags and "ai" in tags:
        return f"{display}: Stanford AI circuit"
    if top_terms:
        return f"{display}: {top_terms[0]}"
    return f"{display}: archetype cluster"


def describe_cluster(anchor: str, tag_counts: Counter[str], employers: list[dict[str, Any]], schools: list[dict[str, Any]]) -> str:
    tags = ", ".join(tag for tag, _count in tag_counts.most_common(6))
    employer_text = ", ".join(item["value"] for item in employers[:4])
    school_text = ", ".join(item["value"] for item in schools[:3])
    parts = [f"{label_display(anchor)} cluster"]
    if tags:
        parts.append(f"signals: {tags}")
    if employer_text:
        parts.append(f"employers: {employer_text}")
    if school_text:
        parts.append(f"schools: {school_text}")
    return "; ".join(parts) + "."


def main() -> None:
    args = parse_args()
    dossier_path = Path(args.dossiers)
    embedding_path = Path(args.embeddings)
    out_dir = Path(args.out_dir)
    dossiers = read_jsonl(dossier_path)
    data = np.load(embedding_path, allow_pickle=True)
    X = np.asarray(data["embeddings"], dtype=np.float32)
    if len(dossiers) != X.shape[0]:
        raise SystemExit(f"Dossier count ({len(dossiers)}) does not match embedding rows ({X.shape[0]}).")

    labels = sorted({record.get("primaryLabel") or "unlabeled" for record in dossiers})
    print(
        json.dumps(
            {
                "dossiers": str(dossier_path),
                "embeddings": str(embedding_path),
                "shape": list(X.shape),
                "labels": {label: sum(1 for record in dossiers if (record.get("primaryLabel") or "unlabeled") == label) for label in labels},
                "clustersPerLabel": args.clusters_per_label,
                "outDir": str(out_dir),
                "dryRun": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments: list[dict[str, Any]] = [{} for _ in dossiers]
    clusters: list[dict[str, Any]] = []

    for label in labels:
        indices = np.asarray([index for index, record in enumerate(dossiers) if (record.get("primaryLabel") or "unlabeled") == label])
        if len(indices) == 0:
            continue
        k = min(args.clusters_per_label, max(1, int(math.sqrt(len(indices)))))
        if k <= 1:
            local_labels = np.zeros(len(indices), dtype=np.int64)
            centers = np.asarray([X[indices].mean(axis=0)])
        else:
            model = KMeans(n_clusters=k, random_state=args.seed, n_init="auto")
            local_labels = model.fit_predict(X[indices])
            centers = model.cluster_centers_

        for local_cluster in range(k):
            cluster_indices = indices[local_labels == local_cluster]
            if len(cluster_indices) == 0:
                continue
            cluster_id = f"{label}_{local_cluster:02d}"
            distances = cosine_distance_to_centroid(X, cluster_indices, centers[local_cluster])
            rep_order = np.argsort(distances)[: args.representatives]
            representative_indices = cluster_indices[rep_order]
            cluster_records = [dossiers[int(index)] for index in cluster_indices]
            representative_records = [dossiers[int(index)] for index in representative_indices]
            tag_counts = Counter(tag for record in cluster_records for tag in (record.get("careerTags") or []))
            employers = top_counts([employer for record in cluster_records for employer in (record.get("employers") or [])], 12)
            schools = top_counts([school for record in cluster_records for school in (record.get("schools") or [])], 10)
            skills = top_counts([skill for record in cluster_records for skill in (record.get("skills") or [])], 14)
            titles = top_counts([title for record in cluster_records for title in (record.get("titles") or [])], 14)
            terms = terms_for_cluster([record.get("dossierText", "") for record in cluster_records], args.top_terms)

            name = cluster_name(label, tag_counts, terms, len(cluster_indices))
            cluster = {
                "clusterId": cluster_id,
                "anchorLabel": label,
                "name": name,
                "description": describe_cluster(label, tag_counts, employers, schools),
                "size": int(len(cluster_indices)),
                "topTerms": terms,
                "careerTags": [{"value": value, "count": count} for value, count in tag_counts.most_common(16)],
                "topEmployers": employers,
                "topSchools": schools,
                "topTitles": titles,
                "topSkills": skills,
                "representatives": [
                    {
                        "rowIndex": int(index),
                        "name": record.get("name", ""),
                        "sourceKey": record.get("sourceKey", ""),
                        "archetypeLabels": record.get("archetypeLabels") or [],
                        "careerTags": record.get("careerTags") or [],
                        "summary": record.get("summary", ""),
                    }
                    for index, record in zip(representative_indices, representative_records)
                ],
            }
            clusters.append(cluster)

            for index in cluster_indices:
                record = dossiers[int(index)]
                assignments[int(index)] = {
                    "rowIndex": int(index),
                    "name": record.get("name", ""),
                    "sourceKey": record.get("sourceKey", ""),
                    "primaryLabel": record.get("primaryLabel", ""),
                    "archetypeLabels": record.get("archetypeLabels") or [],
                    "clusterId": cluster_id,
                    "clusterName": name,
                }

    slug = slug_from_path(embedding_path)
    clusters_path = out_dir / f"{slug}_clusters.json"
    assignments_path = out_dir / f"{slug}_cluster_assignments.jsonl"
    summary_path = out_dir / f"{slug}_clusters_summary.json"

    clusters.sort(key=lambda cluster: (cluster["anchorLabel"], cluster["clusterId"]))
    clusters_path.write_text(json.dumps({"clusters": clusters}, indent=2), encoding="utf-8")
    with assignments_path.open("w", encoding="utf-8") as handle:
        for assignment in assignments:
            handle.write(json.dumps(assignment, ensure_ascii=False) + "\n")
    summary = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dossiers": str(dossier_path.resolve()),
        "embeddings": str(embedding_path.resolve()),
        "clusters": len(clusters),
        "records": len(dossiers),
        "clustersPerLabel": args.clusters_per_label,
        "elapsedSeconds": round(time.time() - started, 3),
        "outputs": {
            "clusters": str(clusters_path),
            "assignments": str(assignments_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(clusters_path), "assignments": str(assignments_path), "clusters": len(clusters)}, indent=2))


if __name__ == "__main__":
    main()
