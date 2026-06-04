#!/usr/bin/env python3
"""Download primary profile images from a Diffbot Person JSONL snapshot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass


USER_AGENT = "FounderModeResearch/0.1 (+https://github.com/leannchen86/founder-mode)"
IMAGE_MAGIC = {
    b"\xff\xd8\xff": ".jpg",
    b"\x89PNG\r\n\x1a\n": ".png",
    b"GIF87a": ".gif",
    b"GIF89a": ".gif",
    b"RIFF": ".webp",
}
CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/heic": ".heic",
    "image/heif": ".heif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/raw/diffbot_people_latest.jsonl",
        help="Diffbot Person JSONL snapshot.",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/fast1/leann/founder-mode/images",
        help="Directory where images will be stored.",
    )
    parser.add_argument(
        "--manifest",
        default="/mnt/fast1/leann/founder-mode/images_manifest.jsonl",
        help="JSONL manifest path for download results.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download workers.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-bytes", type=int, default=15_000_000, help="Maximum image size in bytes.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max people to process.")
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Download every image URL in each record instead of only the primary image.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned download counts only.")
    return parser.parse_args()


def read_people(path: Path, limit: int) -> list[dict[str, Any]]:
    people: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            people.append(json.loads(line))
            if limit and len(people) >= limit:
                break
    return people


def label_for(person: dict[str, Any]) -> str:
    labels = person.get("archetypeLabels") or ["unlabeled"]
    return str(labels[0]).replace("/", "_")


def image_urls(person: dict[str, Any], include_all: bool) -> list[str]:
    if include_all:
        urls = []
        for url in [person.get("image"), *(person.get("images") or [])]:
            if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in urls:
                urls.append(url)
        return urls

    url = person.get("image")
    return [url] if isinstance(url, str) and url.startswith(("http://", "https://")) else []


def stable_key(person: dict[str, Any], url: str, index: int) -> str:
    raw = "|".join(
        [
            str(person.get("diffbotUri") or ""),
            str(person.get("id") or ""),
            str(person.get("linkedInUri") or ""),
            str(url),
            str(index),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def extension_from_url(url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ""


def extension_from_magic(data: bytes) -> str:
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return ".webp"
    for magic, extension in IMAGE_MAGIC.items():
        if data.startswith(magic):
            return extension
    return ""


def planned_downloads(people: list[dict[str, Any]], include_all: bool, out_dir: Path) -> list[dict[str, Any]]:
    jobs = []
    for person in people:
        label = label_for(person)
        for index, url in enumerate(image_urls(person, include_all)):
            key = stable_key(person, url, index)
            extension = extension_from_url(url) or ".img"
            path = out_dir / label / f"{key}{extension}"
            jobs.append(
                {
                    "person": person,
                    "label": label,
                    "index": index,
                    "url": url,
                    "key": key,
                    "path": path,
                }
            )
    return jobs


def response_extension(content_type: str, data: bytes, fallback_path: Path) -> str:
    normalized = content_type.split(";")[0].strip().lower()
    return CONTENT_TYPE_EXTENSIONS.get(normalized) or extension_from_magic(data) or fallback_path.suffix or ".img"


def final_path(job: dict[str, Any], content_type: str, data: bytes) -> Path:
    extension = response_extension(content_type, data, job["path"])
    return job["path"].with_suffix(extension)


def existing_path(job: dict[str, Any]) -> Path | None:
    directory = job["path"].parent
    stem = job["path"].stem
    for path in directory.glob(f"{stem}.*"):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def download_one(job: dict[str, Any], timeout: float, max_bytes: int) -> dict[str, Any]:
    person = job["person"]
    existing = existing_path(job)
    base_record = {
        "status": "pending",
        "source": "diffbot",
        "personId": person.get("id") or "",
        "diffbotUri": person.get("diffbotUri") or "",
        "name": person.get("name") or "",
        "archetypeLabels": person.get("archetypeLabels") or [],
        "imageIndex": job["index"],
        "url": job["url"],
        "key": job["key"],
    }
    if existing:
        return {
            **base_record,
            "status": "skipped_existing",
            "path": str(existing),
            "bytes": existing.stat().st_size,
        }

    request = urllib.request.Request(job["url"], headers={"User-Agent": USER_AGENT})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                return {
                    **base_record,
                    "status": "failed",
                    "error": f"content-length exceeds max-bytes: {content_length}",
                }
            data = response.read(max_bytes + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {**base_record, "status": "failed", "error": str(error)}

    if len(data) > max_bytes:
        return {**base_record, "status": "failed", "error": "download exceeds max-bytes"}

    extension = response_extension(content_type, data, job["path"])
    if extension == ".img":
        return {
            **base_record,
            "status": "failed",
            "error": f"not a recognized image: content-type={content_type!r}",
        }

    path = final_path(job, content_type, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    return {
        **base_record,
        "status": "downloaded",
        "path": str(path),
        "bytes": len(data),
        "sha256": digest,
        "contentType": content_type,
        "elapsedMs": round((time.time() - started) * 1000),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    people = read_people(input_path, args.limit)
    jobs = planned_downloads(people, args.all_images, out_dir)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "outDir": str(out_dir),
                "manifest": str(manifest_path),
                "people": len(people),
                "downloads": len(jobs),
                "allImages": args.all_images,
                "dryRun": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with manifest_path.open("a", encoding="utf-8") as manifest:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_one, job, args.timeout, args.max_bytes) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                counts[record["status"]] = counts.get(record["status"], 0) + 1
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()

    print(json.dumps({"counts": counts}, indent=2))


if __name__ == "__main__":
    main()
