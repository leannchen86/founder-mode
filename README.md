# Founder Mode Data

Small local data workspace for the foundermode.com fortune-teller prototype.

## GitHub Pages Prototype

The static prototype lives in `site/` and is deployable with GitHub Pages. It can run the camera UI and demo oracle copy in the browser, but GitHub Pages cannot run the local Python/MetaCLIP retrieval pipeline. The live model path should be connected later through a backend API.

Run it locally:

```bash
python3 -m http.server 4173 --directory site
```

After pushing to `main`, `.github/workflows/deploy-pages.yml` uploads `site/` and deploys it with GitHub Pages Actions. The expected project URL is:

```text
https://leannchen86.github.io/founder-mode/
```

The first crawler pulls Bay Area `Person` records from Diffbot for three starter archetype buckets:

- `founders`
- `vcs`
- `faang_engineers`

It intentionally collects profile metadata and image URLs only. It does not request or store phone numbers or email addresses.

Token lookup order:

1. `DIFFBOT_TOKEN` in the shell
2. `foundermode/.env.local`
3. `foundermode/.env`
4. `../chindian-heatmap/.env.local`

Inspect DQL without spending credits:

```bash
node scripts/fetch-diffbot-people.mjs --dry-run
```

Count available Bay Area profile pools:

```bash
node scripts/count-diffbot-people.mjs
```

Run a small starter crawl:

```bash
node scripts/fetch-diffbot-people.mjs --max-per-class=25 --page-size=10
```

Summarize the latest snapshot:

```bash
node scripts/summarize-diffbot-people.mjs
```

Download profile images, usually on Zeus fast storage:

```bash
python3 scripts/download-profile-images.py \
  --input data/raw/diffbot_people_latest.jsonl \
  --out-dir /mnt/fast1/leann/founder-mode/images \
  --manifest /mnt/fast1/leann/founder-mode/images_manifest.jsonl \
  --workers 12
```

By default this downloads only each person's primary profile image. Add `--all-images` only if you want every image URL from the record.

From the local machine, sync the current workspace to Zeus and run the downloader:

```bash
bash scripts/run-zeus-image-download.sh
```

For a smoke run:

```bash
LIMIT=5 bash scripts/run-zeus-image-download.sh
```

Clean downloaded local images into RetinaFace crops:

```bash
.venv/bin/python scripts/clean-face-images.py \
  --manifest data/images/raw_manifest.jsonl \
  --out-dir data/images/clean \
  --clean-manifest data/images/clean_manifest.jsonl \
  --gpu-id -1
```

Extract frozen CLIP embeddings from clean crops:

```bash
.venv/bin/python scripts/extract-image-embeddings.py \
  --clean-manifest data/images/clean_manifest.jsonl \
  --out-dir data/embeddings \
  --device auto
```

For a MetaCLIP run with the matching QuickGELU ViT-L/14 architecture:

```bash
.venv/bin/python scripts/extract-image-embeddings.py \
  --clean-manifest data/images/clean_manifest.jsonl \
  --out-dir data/embeddings \
  --model ViT-L-14-quickgelu \
  --pretrained metaclip_fullcc \
  --device auto
```

Extract frozen DINOv3 embeddings from the same clean crops:

```bash
.venv/bin/python scripts/extract-dino-image-embeddings.py \
  --clean-manifest data/images/clean_manifest.jsonl \
  --out-dir data/embeddings \
  --model vit_base_patch16_dinov3 \
  --device auto
```

The default DINOv3 command uses the public `timm` backbone. Official `facebook/dinov3-*` Hugging Face checkpoints can also be used with `--backend transformers` after authenticating with Hugging Face and accepting the model terms.

Evaluate image embeddings and generate visual QA sheets:

```bash
.venv/bin/python scripts/evaluate-image-embeddings.py \
  --embeddings data/embeddings/ViT-L-14_openai_image_embeddings.npz \
  --out-dir data/eval
```

For the default DINOv3 output:

```bash
.venv/bin/python scripts/evaluate-image-embeddings.py \
  --embeddings data/embeddings/vit_base_patch16_dinov3_cls_image_embeddings.npz \
  --out-dir data/eval
```

Build profile text dossiers from Diffbot metadata:

```bash
node scripts/build-profile-dossiers.mjs \
  --input=data/raw/diffbot_people_latest.jsonl \
  --output=data/processed/profile_dossiers_latest.jsonl
```

Embed those profile dossiers locally:

```bash
.venv/bin/python scripts/embed-profile-text.py \
  --input data/processed/profile_dossiers_latest.jsonl \
  --out-dir data/text_embeddings
```

Cluster the text embeddings into founder/VC/big-tech archetypes:

```bash
.venv/bin/python scripts/cluster-profile-text.py \
  --dossiers data/processed/profile_dossiers_latest.jsonl \
  --embeddings data/text_embeddings/sentence-transformers-all-MiniLM-L6-v2_mean_profile_text_embeddings.npz \
  --out-dir data/archetypes
```

Join cleaned face/image rows to their text archetypes:

```bash
.venv/bin/python scripts/link-image-text-archetypes.py \
  --image-embeddings data/embeddings/ViT-L-14-quickgelu_metaclip_fullcc_image_embeddings.npz \
  --raw-manifest data/images/raw_1k_manifest.jsonl \
  --cluster-assignments data/archetypes/sentence-transformers-all-MiniLM-L6-v2_mean_cluster_assignments.jsonl \
  --clusters data/archetypes/sentence-transformers-all-MiniLM-L6-v2_mean_clusters.json \
  --out-dir data/archetypes
```

Run the first multimodal Founder Mode oracle on an existing embedding row:

```bash
.venv/bin/python scripts/run-founder-oracle.py \
  --name "Reid Hoffman" \
  --top-k 40
```

Oracle output includes both a stricter theatrical `founderModePercent` and an `auraMix` such as `44% founder / 39% VC / 17% big-tech`. By default, scores below `65` or weak top visual matches return the product verdict `No archetype detected`.

Or run it on a face image/crop:

```bash
.venv/bin/python scripts/run-founder-oracle.py \
  --image data/images/clean_1k/founders/d36d6a8f225c80794a6757c2.jpg \
  --top-k 40
```

Crawl outputs land in `data/raw/`:

- `diffbot_people_<timestamp>.jsonl`
- `diffbot_people_<timestamp>.summary.json`
- `diffbot_people_latest.jsonl`
- `diffbot_people_latest.summary.json`

Image downloads, clean crops, embeddings, eval reports, text archetypes, and oracle demos land in `data/images/`, `data/embeddings/`, `data/text_embeddings/`, `data/eval/`, `data/archetypes/`, and `data/oracle/`.
