#!/usr/bin/env python3
"""Join clean face/image embedding rows to text archetype assignments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-embeddings", default="data/embeddings/ViT-L-14-quickgelu_metaclip_fullcc_image_embeddings.npz")
    parser.add_argument("--raw-manifest", default="data/images/raw_1k_manifest.jsonl")
    parser.add_argument("--cluster-assignments", default="data/archetypes/sentence-transformers-all-MiniLM-L6-v2_mean_cluster_assignments.jsonl")
    parser.add_argument("--clusters", default="data/archetypes/sentence-transformers-all-MiniLM-L6-v2_mean_clusters.json")
    parser.add_argument("--out-dir", default="data/archetypes")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def slug_from_path(path: Path) -> str:
    return path.stem.removesuffix("_image_embeddings")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_embeddings)
    assignment_path = Path(args.cluster_assignments)
    clusters_path = Path(args.clusters)
    raw_manifest_path = Path(args.raw_manifest)
    out_dir = Path(args.out_dir)

    image_data = np.load(image_path, allow_pickle=True)
    source_keys = [str(value) for value in image_data["source_keys"].tolist()]
    names = [str(value) for value in image_data["names"].tolist()]
    paths = [str(value) for value in image_data["paths"].tolist()] if "paths" in image_data.files else [""] * len(source_keys)
    labels = np.asarray(image_data["labels"], dtype=np.int64) if "labels" in image_data.files else np.zeros(len(source_keys), dtype=np.int64)
    label_names = [str(value) for value in image_data["label_names"].tolist()] if "label_names" in image_data.files else ["unlabeled"]

    assignment_records = read_jsonl(assignment_path)
    assignments = {record.get("sourceKey", ""): record for record in assignment_records}
    assignments_by_name = {str(record.get("name", "")).strip().lower(): record for record in assignment_records if record.get("name")}
    raw_records = read_jsonl(raw_manifest_path) if raw_manifest_path.exists() else []
    raw_by_image_key = {record.get("key", ""): record for record in raw_records}
    clusters_payload = json.loads(clusters_path.read_text(encoding="utf-8"))
    clusters = {cluster["clusterId"]: cluster for cluster in clusters_payload["clusters"]}

    print(
        json.dumps(
            {
                "imageEmbeddings": str(image_path),
                "imageRows": len(source_keys),
                "clusterAssignments": str(assignment_path),
                "assignments": len(assignments),
                "rawManifest": str(raw_manifest_path),
                "rawRows": len(raw_records),
                "clusters": len(clusters),
                "outDir": str(out_dir),
                "dryRun": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{slug_from_path(image_path)}__{assignment_path.stem.removesuffix('_cluster_assignments')}"
    links_path = out_dir / f"{slug}_image_text_links.jsonl"
    summary_path = out_dir / f"{slug}_image_text_links_summary.json"

    matched = 0
    cluster_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for row_index, source_key in enumerate(source_keys):
        raw = raw_by_image_key.get(source_key, {})
        candidates = [
            source_key,
            raw.get("diffbotUri", ""),
            raw.get("personId", ""),
            names[row_index].strip().lower(),
        ]
        assignment = next((assignments[candidate] for candidate in candidates if candidate in assignments), None)
        if assignment is None:
            assignment = assignments_by_name.get(names[row_index].strip().lower())
        image_label = label_names[int(labels[row_index])] if int(labels[row_index]) < len(label_names) else "unlabeled"
        label_counts[image_label] += 1
        row = {
            "imageRowIndex": row_index,
            "sourceKey": source_key,
            "name": names[row_index],
            "imagePath": paths[row_index],
            "imageLabel": image_label,
            "downloadKey": source_key,
            "diffbotUri": raw.get("diffbotUri", ""),
            "personId": raw.get("personId", ""),
            "matchedTextArchetype": bool(assignment),
        }
        if assignment:
            matched += 1
            cluster = clusters.get(assignment["clusterId"], {})
            cluster_counts[assignment["clusterId"]] += 1
            row.update(
                {
                    "textPrimaryLabel": assignment.get("primaryLabel", ""),
                    "textArchetypeLabels": assignment.get("archetypeLabels") or [],
                    "clusterId": assignment.get("clusterId", ""),
                    "clusterName": assignment.get("clusterName", ""),
                    "clusterAnchorLabel": cluster.get("anchorLabel", ""),
                    "clusterDescription": cluster.get("description", ""),
                    "clusterCareerTags": cluster.get("careerTags", [])[:8],
                }
            )
        rows.append(row)

    with links_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    top_clusters = []
    for cluster_id, count in cluster_counts.most_common():
        cluster = clusters.get(cluster_id, {})
        top_clusters.append(
            {
                "clusterId": cluster_id,
                "name": cluster.get("name", ""),
                "anchorLabel": cluster.get("anchorLabel", ""),
                "count": count,
            }
        )

    summary = {
        "imageEmbeddings": str(image_path.resolve()),
        "rawManifest": str(raw_manifest_path.resolve()) if raw_manifest_path.exists() else "",
        "clusterAssignments": str(assignment_path.resolve()),
        "clusters": str(clusters_path.resolve()),
        "imageRows": len(source_keys),
        "matched": matched,
        "unmatched": len(source_keys) - matched,
        "imageLabelCounts": dict(label_counts),
        "topLinkedClusters": top_clusters,
        "outputs": {
            "links": str(links_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(links_path), "matched": matched, "unmatched": len(source_keys) - matched}, indent=2))


if __name__ == "__main__":
    main()
