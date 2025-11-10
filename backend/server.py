from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()

# ✅ อนุญาตให้ frontend เข้าถึง backend ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ โหลดโมเดล custom (18 classes)
model = YOLO("best.pt")
CONFIDENCE_THRESHOLD = 0.4

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # อ่านภาพ
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # ตรวจจับวัตถุ
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        color = (0, 255, 0)

        # วาดกรอบและข้อความ
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        detections.append({
            "class": label,
            "confidence": round(conf, 2)
        })

    # แปลงภาพเป็น base64 ส่งกลับ
    _, buffer = cv2.imencode(".jpg", frame)
    encoded_img = base64.b64encode(buffer).decode("utf-8")

    return {"image": encoded_img, "detections": detections}
