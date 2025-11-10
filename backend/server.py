from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
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

# 🧠 กฎสำหรับเมนูอาหารไทย (ใช้ชื่อ class เป็นอังกฤษทั้งหมด)
MENU_RULES = [
    {"menu": "ข้าวมันไก่ต้ม", "must_have": ["chicken_rice", "boiled_chicken", "rice"], "optional": ["boiled_chicken_blood_jelly", "cucumber"]},
    {"menu": "ข้าวมันไก่ทอด", "must_have": ["chicken_rice", "fried_chicken", "rice"], "optional": ["cucumber"]},
    {"menu": "ข้าวมันไก่ทอดไก่ต้ม", "must_have": ["chicken_rice", "fried_chicken", "boiled_chicken", "rice"], "optional": ["boiled_chicken_blood_jelly", "cucumber"]},
    {"menu": "ข้าวหมูแดงหมูกรอบ", "must_have": ["red_pork_and_crispy_pork", "rice", "chainese_sausage", "crispy_pork", "boiled_egg"], "optional": ["cucumber"]},
    {"menu": "ก๋วยเตี๋ยวไก่น่องตุ๋นยาจีน", "must_have": ["noodle", "chicken_drumstick"], "optional": ["boiled_chicken_blood_jelly", "daikon_radish"]},
    {"menu": "ก๋วยเตี๋ยวไก่ฉีกตุ๋นยาจีน", "must_have": ["noodle", "chicken_shredded"], "optional": ["boiled_chicken_blood_jelly", "daikon_radish"]},
    {"menu": "ข้าวกะเพราหมูสับ", "must_have": ["stir_fried_basil", "minced_pork", "rice"], "optional": []},
    {"menu": "ข้าวกะเพราหมูสับเต้าหู้", "must_have": ["stir_fried_basil", "minced_pork", "rice", "fried_tofo"], "optional": []},
]

# ✅ โหลดโมเดล
try:
    model = YOLO("models/best.pt")
    print("✅ โหลดโมเดลสำเร็จ")
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
        results = model(image, conf=0.25)[0]
    except Exception as e:
        return {"error": f"Inference failed: {e}"}

    detections = []
    draw = ImageDraw.Draw(image)

    # ✅ ฟอนต์ใหญ่ขึ้น
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = None

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names.get(cls, f"class_{cls}")

            draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
            label = f"{class_name} {conf*100:.1f}%"
            draw.text((x1, max(0, y1 - 30)), label, fill="lime", font=font)

            detections.append({
                "class": class_name,
                "confidence": conf
            })
    else:
        print("⚠️ ไม่มีข้อมูลการตรวจจับ")

    # ✅ ตรวจว่าตรงกับเมนูไหน
    detected_classes = [d["class"] for d in detections]
    matched_menu = None
    matched_components = []

    for rule in MENU_RULES:
        if all(item in detected_classes for item in rule["must_have"]):
            matched_menu = rule["menu"]
            matched_components = rule["must_have"] + [x for x in rule["optional"] if x in detected_classes]
            break

    # 🔄 แปลงภาพเป็น Base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode()

    # ✅ ส่งผลลัพธ์กลับ
    if matched_menu:
        components_info = []
        for comp in matched_components:
            conf = next((d["confidence"] for d in detections if d["class"] == comp), None)
            components_info.append({
                "name": comp,
                "confidence": round(conf * 100, 1) if conf else None
            })

        return {
            "image": encoded_img,
            "predicted_menu": matched_menu,
            "components": components_info
        }
    else:
        return {
            "image": encoded_img,
            "predicted_menu": "ไม่พบเมนูที่ตรง",
            "detections": [
                {"name": d["class"], "confidence": round(d["confidence"] * 100, 1)}
                for d in detections
            ]
        }
