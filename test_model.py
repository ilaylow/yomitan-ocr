from rapidocr_onnxruntime import RapidOCR
from huggingface_hub import hf_hub_download
import cv2

def draw_boxes_cv(image_path, ocr_result, out_path="boxed_cv.png"):
    img = cv2.imread(image_path)

    for line in ocr_result:
        box = line[0]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]

        pt1 = (int(min(xs)), int(min(ys)))
        pt2 = (int(max(xs)), int(max(ys)))

        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)

    cv2.imwrite(out_path, img)

det_path = "/Users/chuenleylow/Downloads/onnx-test/ppocrv5_det.onnx"
rec_path = "/Users/chuenleylow/Downloads/onnx-test/ppocrv5_rec.onnx"
dict_path = hf_hub_download("monkt/paddleocr-onnx", "languages/chinese/dict.txt")

ocr = RapidOCR(
    det_model_path=det_path,
    rec_model_path=rec_path,
    rec_keys_path=dict_path
)

result, elapsed = ocr("document.png")
for res in result:
    print(f"Word: {res[1]} with Probability: {res[2]}")

draw_boxes_cv("document.png", result)
