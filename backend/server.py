from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image, ImageDraw
import base64
from ultralytics import YOLO

app = FastAPI()

# ✅ อนุญาตให้ frontend เข้าถึง
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# แปล Class เป็นภาษาไทย
CLASS_TRANSLATIONS = {
    "boiled_chicken": "ไก่ต้ม",
    "boiled_chicken_blood_jelly": "เลือดไก่ต้ม",
    "boiled_egg": "ไข่ต้ม",
    "chainese_sausage": "กุนเชียง",
    "chicken_drumstick": "น่องไก่",
    "chicken_rice": "ข้าวมันไก่",
    "chicken_shredded": "ไก่ฉีก",
    "crispy_pork": "หมูกรอบ",
    "cucumber": "แตงกวา",
    "daikon_radish": "ไชเท้า",
    "fried_chicken": "ไก่ทอด",
    "fried_tofo": "เต้าหู้ทอด",
    "minced_pork": "หมูสับ",
    "noodle": "ก๋วยเตี๋ยว",
    "red_pork": "หมูแดง",
    "red_pork_and_crispy_pork": "ข้าวหมูแดงหมูกรอบ",
    "rice": "ข้าว",
    "stir_fried_basil": "กะเพรา",
}

# ✅ โหลดโมเดล
try:
    model = YOLO("models/best.pt")
except Exception as e:
    print("❌ โหลดโมเดลไม่สำเร็จ:", e)
    model = None

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image: {e}"}

    # 🔍 ตรวจจับ
    try:
        results = model(image)[0]
    except Exception as e:
        return {"error": f"Inference failed: {e}"}

    detections = []
    draw = ImageDraw.Draw(image)

    # 🔸 ป้องกันผลลัพธ์ว่าง
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names.get(cls, f"class_{cls}")
            thai_name = CLASS_TRANSLATIONS.get(class_name, class_name)


            # วาดกรอบ
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            draw.text((x1, y1 - 10), f"{class_name} {conf:.2f}", fill="lime")

            detections.append({
                "class_name": thai_name,
                "confidence": conf
            })

    # แปลงภาพเป็น Base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode()

    return {"image": encoded_img, "detections": detections}
