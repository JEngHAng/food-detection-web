from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io

app = FastAPI()

# CORS (อนุญาตให้ frontend เรียก API ได้)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล (ใช้ yolov5 หรือ yolov8 ก็ได้)
model = torch.hub.load("ultralytics/yolov5", "custom", path="models/best.pt", force_reload=False)
model.conf = 0.25  # ค่าความมั่นใจขั้นต่ำ

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # อ่านไฟล์ภาพ
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # ใช้โมเดลตรวจจับ
        results = model(image)
        detections = results.pandas().xyxy[0].to_dict(orient="records")

        if not detections:
            return JSONResponse({
                "menu": [],
                "components": [],
                "message": "No matching menu found"
            })

        # จัดข้อมูล
        components = []
        menus = set()

        for det in detections:
            label = det["name"]
            conf = round(det["confidence"] * 100, 1)
            components.append({"label": label, "confidence": conf})
            menus.add(label.split("_")[0])  # เช่น "chicken_rice" จาก "chicken_rice_meat"

        return JSONResponse({
            "menu": list(menus),
            "components": components,
            "message": "Multiple detections found"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
