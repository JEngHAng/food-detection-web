from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = FastAPI()

# ✅ อนุญาตให้ frontend เข้ามาเรียกใช้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ โหลดโมเดล YOLO รุ่นใหม่ (Ultralytics)
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

# ✅ mapping ชื่อ class → ชื่อเมนูจริง
MENU_MAP = {
    "chicken_rice": "Khao Man Gai (Chicken Rice)",
    "fried_chicken": "Fried Chicken",
    "boiled_chicken": "Boiled Chicken",
    "cucumber": "Cucumber",
    "red_pork_and_crispy_pork": "Red Pork & Crispy Pork Rice",
    "boiled_chicken_blood_jelly": "Boiled Chicken Blood Jelly",
}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # ✅ อ่านภาพจาก frontend
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ✅ รันโมเดล (YOLOv8)
        results = model.predict(img, imgsz=640, conf=0.25)
        detections = results[0].boxes.data.cpu().numpy() if results else []

        components = []
        seen_menus = set()

        # ✅ วาด bounding box และเก็บข้อมูล
        img_cv = np.array(img)
        for *box, conf, cls in detections:
            name = model.names[int(cls)]
            confidence = round(float(conf) * 100, 1)

            # ✅ แปลงชื่อ class เป็นชื่อเมนูจริง
            menu_name = MENU_MAP.get(name, name)
            components.append({"name": menu_name, "confidence": confidence})

            # ✅ ตรวจเมนูหลัก (หากเจอหลายอย่าง)
            seen_menus.add(menu_name)

            # ✅ วาดกรอบรอบวัตถุ
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(
                img_cv,
                f"{menu_name} {confidence:.1f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 100, 0),
                2,
            )

        # ✅ แปลงภาพกลับเป็น base64 เพื่อส่งกลับ
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        if len(components) == 0:
            return JSONResponse(
                content={
                    "predicted_menus": [],
                    "detections": [],
                    "image": image_base64,
                    "error": "No objects detected",
                }
            )

        # ✅ ถ้ามีหลายเมนู ให้รวมทั้งหมด
        return JSONResponse(
            content={
                "predicted_menus": list(seen_menus),
                "detections": components,
                "image": image_base64,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)})


@app.get("/")
def root():
    return {"message": "🍛 Thai Food Detection API is running with YOLOv8 🚀"}
