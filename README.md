---
title: Asset Extractor API
emoji: 🎨
colorFrom: yellow
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Visual Asset Extractor API

FastAPI backend for extracting visual assets from presentation slides using Grounding DINO + flood-fill background removal.

## Endpoints

- `GET /health` — Health check
- `POST /extract` — Extract assets from a single image
- `POST /extract-pdf` — Extract assets from all pages of a PDF

## Deploy to HF Spaces

1. Create a new HF Space with **Docker** SDK
2. Push this repo to the Space
3. It will build and deploy automatically

## Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```
