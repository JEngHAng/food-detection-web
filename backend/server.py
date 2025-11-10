from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Allow all CORS (ให้ frontend ติดต่อได้)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล YOLOv8
model = YOLO("models/best.pt")  # ใส่ path โมเดลของคุณ

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # อ่านรูปจาก frontend
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # รันตรวจจับ
    results = model(image)

    # ดึงผลลัพธ์
    components = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]
        conf = float(box.conf)
        components.append({
            "label": label,
            "confidence": round(conf * 100, 1)
        })

    # ตรงนี้คือ “จับคู่เมนู” แบบง่าย (เช่น ถ้ามีของครบถึงเรียกว่าเมนูนั้น)
    # สามารถแก้ mapping ตามที่คุณเคยใช้เองได้
    menu_rules = {
        "Khao Man Gai (Chicken Rice)": ["boiled_chicken", "cucumber", "boiled_chicken_blood_jelly"],
        "Fried Chicken Rice": ["fried_chicken", "cucumber"],
        "Red Pork & Crispy Pork Rice": ["red_pork_and_crispy_pork", "cucumber"]
    }

    detected_labels = [c["label"] for c in components]
    detected_menus = []

    for menu_name, ingredients in menu_rules.items():
        if all(item in detected_labels for item in ingredients):
            detected_menus.append(menu_name)

    if not detected_menus:
        detected_menus = ["No matching menu"]

    return {"menu": detected_menus, "components": components}
