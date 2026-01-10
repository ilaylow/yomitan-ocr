from fastapi import FastAPI, UploadFile, File, HTTPException
from rapidocr_onnxruntime import RapidOCR
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
import numpy as np
import cv2

app = FastAPI(title="OCR API", description="API for detecting text in images")

# Model paths
det_path = "/Users/chuenleylow/Downloads/onnx-test/ppocrv5_det.onnx"
rec_path = "/Users/chuenleylow/Downloads/onnx-test/ppocrv5_rec.onnx"
dict_path = hf_hub_download("monkt/paddleocr-onnx", "languages/chinese/dict.txt")

# Initialize OCR model
ocr = RapidOCR(
    det_model_path=det_path,
    rec_model_path=rec_path,
    rec_keys_path=dict_path
)


class DetectedWord(BaseModel):
    text: str
    confidence: float


class OCRResponse(BaseModel):
    words: list[DetectedWord]


@app.post("/ocr", response_model=OCRResponse)
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

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result, elapsed = ocr(img)
    print(f"Detection: {elapsed[0]:.3f}s, Classification: {elapsed[1]:.3f}s, Recognition: {elapsed[2]:.3f}s")

    if result is None:
        return OCRResponse(words=[])

    words = [
        DetectedWord(text=res[1], confidence=round(float(res[2]) * 100, 2))
        for res in result
    ]

    return OCRResponse(words=words)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
