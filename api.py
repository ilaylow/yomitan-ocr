from fastapi import FastAPI, UploadFile, File, HTTPException
from rapidocr_onnxruntime import RapidOCR
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
from manga_ocr import MangaOcr
from PIL import Image
import io

app = FastAPI(title="OCR API", description="API for detecting text in images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:xxxx"]
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
det_path = "/Users/chuenleylow/Downloads/onnx-test/ppocrv5_det.onnx"
rec_path = "/Users/chuenleylow/Downloads/onnx-test/ppocrv5_rec.onnx"
dict_path = hf_hub_download("monkt/paddleocr-onnx", "languages/chinese/dict.txt")

# Initialize OCR model
ocr = RapidOCR(
    det_model_path=det_path,
    rec_model_path=rec_path,
    rec_keys_path=dict_path,
    det_db_box_thresh=0.1,  # lower from default 0.5 - more sensitive detection
    det_db_unclip_ratio=1.8,
)

# Initialize secondary OCR model
mocr = MangaOcr()


class DetectedWord(BaseModel):
    text: str
    confidence: float


class OCRResponse(BaseModel):
    words: list[DetectedWord]


@app.post("/yomitan/ocr/")
async def process_image(file: UploadFile = File(...)):
    """
    Process an uploaded image and return detected text with confidence scores.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()

    # Convert bytes to numpy array for OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    pil_image = Image.open(io.BytesIO(nparr))

    if img is None or pil_image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")


    result, elapsed = ocr(img)
    mocr_result = mocr(pil_image)
    if result is None and mocr_result is None:
        return OCRResponse(words=[])
    
    if elapsed:
        print(f"(Primary Model) Detection: {elapsed[0]:.3f}s, Classification: {elapsed[1]:.3f}s, Recognition: {elapsed[2]:.3f}s")

    if result is not None:
        words = [
            DetectedWord(text=res[1], confidence=round(float(res[2]) * 100, 2))
            for res in result
        ]
        return {"words": words, "type": "primary"}
    else:
        words = mocr_result
        return {"words": [words], "type": "secondary"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
