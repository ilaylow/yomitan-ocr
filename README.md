# yomitan-ocr

A small FastAPI service that does OCR on uploaded images and returns the
recognised text. Built to power the clipboard image lookup in
[yomitan-lite-frontend](https://github.com/ilaylow/yomitan-lite-frontend).

Optimised for Japanese, but the primary model is paddle's general-purpose
detector so it works on other scripts too.

## What it does

Takes an image upload, runs it through two OCR models in parallel:

1. **Primary**: RapidOCR running paddleocr v5 ONNX models (det + rec) for fast
   bounded-box text detection
2. **Secondary**: manga_ocr (a specialised model trained on Japanese manga
   pages) used as a fallback when the primary model finds nothing

Falls back to whichever model returned a result, with a `type` field in the
response so the client knows which one fired.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| POST | `/yomitan/ocr/` | Upload an image, get detected text |
| GET | `/health` | Health check |

Sample response shape:

```json
{
  "words": [
    { "text": "学校", "confidence": 98.4 },
    { "text": "行く", "confidence": 96.1 }
  ],
  "type": "primary"
}
```

## Running it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python api.py
```

The server listens on port 8000.

First start downloads the model weights and `dict.txt` from Hugging Face. After
that everything stays on disk and runs offline.

## Notes

- CPU-only. Runs comfortably on a Raspberry Pi 5 for typical lookup-sized
  images. Large pages will be slower but still tolerable.
- The CORS policy is wide open (`allow_origins=["*"]`) because access control
  is handled by the reverse proxy that sits in front of this service.

## Stack

- FastAPI + uvicorn
- RapidOCR (paddleocr v5 ONNX, via `rapidocr_onnxruntime`)
- manga_ocr (PyTorch, transformer-based)
- OpenCV, Pillow for image handling

## License

ISC
