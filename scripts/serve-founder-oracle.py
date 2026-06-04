#!/usr/bin/env python3
"""Serve the real Founder Mode image-embedding oracle over a small local HTTP API."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_SCRIPT = Path(__file__).with_name("run-founder-oracle.py")

spec = importlib.util.spec_from_file_location("founder_oracle", ORACLE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load oracle module from {ORACLE_SCRIPT}")
founder_oracle = importlib.util.module_from_spec(spec)
sys.modules["founder_oracle"] = founder_oracle
spec.loader.exec_module(founder_oracle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--image-embeddings", default=founder_oracle.DEFAULT_IMAGE_EMBEDDINGS)
    parser.add_argument("--links", default=founder_oracle.DEFAULT_LINKS)
    parser.add_argument("--clusters", default=founder_oracle.DEFAULT_CLUSTERS)
    parser.add_argument("--model", default="ViT-L-14-quickgelu")
    parser.add_argument("--pretrained", default="metaclip_fullcc")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--show-neighbors", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.06)
    parser.add_argument("--min-score", type=int, default=65)
    parser.add_argument("--min-top-similarity", type=float, default=0.62)
    parser.add_argument("--max-body-mb", type=float, default=8)
    return parser.parse_args()


def repo_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def decode_image_payload(payload: dict[str, Any]) -> Image.Image:
    value = str(payload.get("imageDataUrl") or payload.get("imageBase64") or "")
    if not value:
        raise ValueError("Missing imageDataUrl.")
    if "," in value:
        _header, value = value.split(",", 1)
    image_bytes = base64.b64decode(value, validate=True)
    return Image.open(io.BytesIO(image_bytes))


def clean_label(value: str) -> str:
    replacements = {
        "ai": "AI",
        "vc": "VC",
        "vcs": "VC",
        "faang": "FAANG",
        "big tech": "big-tech",
        "big-tech": "big-tech",
        "big_tech": "big-tech",
    }
    normalized = value.strip().lower()
    if normalized in replacements:
        return replacements[normalized]
    words = value.replace("_", " ").strip().split()
    return " ".join(replacements.get(word.lower(), word) for word in words)


def anchor_label(value: str) -> str:
    labels = {
        "founders": "Founder",
        "vcs": "VC",
        "faang_engineers": "Big-Tech",
    }
    return labels.get(value, clean_label(value))


def compact(value: str, limit: int = 86) -> str:
    value = " ".join(str(value).split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def display_title(value: str) -> str:
    short = founder_oracle.archetype_short(value)
    if not short:
        return "Bay Area archetype"
    return clean_label(short[:1].upper() + short[1:])


def signals_from_description(value: str) -> list[str]:
    marker = "signals:"
    if marker not in value:
        return []
    after_marker = value.split(marker, 1)[1]
    before_next = after_marker.split(";", 1)[0]
    return [clean_label(item.strip()) for item in before_next.split(",") if item.strip()]


def neighbor_note(item: dict[str, Any]) -> str:
    anchor = anchor_label(str(item.get("anchorLabel", "")))
    signals = signals_from_description(str(item.get("description", "")))[:3]
    if signals:
        return f"{anchor} signal - {', '.join(signals)}"
    return f"{anchor} signal"


def frontend_oracle(raw: dict[str, Any]) -> dict[str, Any]:
    is_miss = bool(raw.get("noArchetypeDetected"))
    primary = raw.get("primaryArchetype") or {}
    aura_mix = raw.get("auraMix") or []
    anchors = [[item.get("label", ""), int(item.get("percent", 0))] for item in aura_mix]
    signals = [clean_label(item.get("value", "")) for item in (raw.get("careerSignalWeights") or [])[:4]]

    neighbors: list[list[str]] = []
    for item in (raw.get("secondaryArchetypes") or [])[:4]:
        neighbors.append(
            [
                display_title(str(item.get("name", ""))),
                compact(neighbor_note(item)),
            ]
        )

    return {
        "score": int(raw.get("founderModePercent", 0)),
        "title": "No archetype detected" if is_miss else display_title(str(primary.get("name", ""))),
        "miss": is_miss,
        "anchors": anchors,
        "signals": [] if is_miss else signals,
        "neighbors": [] if is_miss else neighbors,
        "fortune": raw.get("fortune", ""),
        "confidence": raw.get("confidence", {}),
        "verdict": raw.get("verdict", ""),
    }


class FounderOracleService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = founder_oracle.choose_device(args.device)
        self.image_embeddings_path = repo_path(args.image_embeddings)
        self.links_path = repo_path(args.links)
        self.clusters_path = repo_path(args.clusters)
        self.payload = founder_oracle.load_image_embeddings(self.image_embeddings_path)
        self.links = founder_oracle.read_jsonl(self.links_path)
        clusters_payload = json.loads(self.clusters_path.read_text(encoding="utf-8"))
        self.clusters = {cluster["clusterId"]: cluster for cluster in clusters_payload["clusters"]}
        if self.payload["X"].shape[0] != len(self.links):
            raise RuntimeError(
                f"Embedding rows ({self.payload['X'].shape[0]}) do not match link rows ({len(self.links)})."
            )

        started = time.time()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            args.model,
            pretrained=args.pretrained,
            device=self.device,
        )
        self.model.eval()
        self.load_seconds = round(time.time() - started, 3)
        self.lock = threading.Lock()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        image = ImageOps.exif_transpose(image).convert("RGB")
        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.lock, torch.inference_mode():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().float().numpy()[0]

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        image = decode_image_payload(payload)
        query_vector = self.encode_image(image)
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        top_k = int(payload.get("topK") or self.args.top_k)
        show_neighbors = int(payload.get("showNeighbors") or self.args.show_neighbors)
        min_score = int(payload.get("minScore") or self.args.min_score)
        min_top_similarity = float(payload.get("minTopSimilarity") or self.args.min_top_similarity)
        neighbor_indices, similarities = founder_oracle.top_neighbor_indices(
            self.payload["X"],
            query_vector,
            top_k,
            exclude_index=-1,
        )
        raw = founder_oracle.aggregate_oracle(
            neighbor_indices,
            similarities,
            self.links,
            self.clusters,
            show_neighbors,
            self.args.temperature,
            min_score,
            min_top_similarity,
        )
        return {
            "status": "ok",
            "latencyMs": round((time.time() - started) * 1000),
            "model": {
                "imageEncoder": f"{self.args.model}/{self.args.pretrained}",
                "device": self.device,
                "rows": int(self.payload["X"].shape[0]),
            },
            "oracle": frontend_oracle(raw),
            "raw": raw,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": f"{self.args.model}/{self.args.pretrained}",
            "device": self.device,
            "rows": int(self.payload["X"].shape[0]),
            "loadSeconds": self.load_seconds,
            "minScore": self.args.min_score,
            "minTopSimilarity": self.args.min_top_similarity,
        }


class OracleRequestHandler(BaseHTTPRequestHandler):
    service: FounderOracleService
    max_body_bytes: int

    def add_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.add_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.add_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/health":
            self.send_json(200, self.service.health())
            return
        self.send_json(404, {"status": "error", "error": "Not found."})

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path != "/analyze":
            self.send_json(404, {"status": "error", "error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                raise ValueError("Missing request body.")
            if length > self.max_body_bytes:
                raise ValueError("Request body is too large.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_json(200, self.service.analyze(payload))
        except Exception as error:
            self.send_json(500, {"status": "error", "error": str(error)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def main() -> None:
    args = parse_args()
    service = FounderOracleService(args)
    OracleRequestHandler.service = service
    OracleRequestHandler.max_body_bytes = int(args.max_body_mb * 1024 * 1024)
    server = ThreadingHTTPServer((args.host, args.port), OracleRequestHandler)
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}",
                **service.health(),
                "status": "ready",
                "ready": True,
            },
            indent=2,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
