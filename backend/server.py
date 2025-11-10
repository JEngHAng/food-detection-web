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

# 🧠 กฎสำหรับเมนูอาหารไทย
MENU_RULES = [
    {"menu": "ข้าวมันไก่ต้ม", "must_have": ["ข้าวมันไก่", "ไก่ต้ม", "ข้าว"], "optional": ["เลือดไก่ต้ม", "แตงกวา"]},
    {"menu": "ข้าวมันไก่ทอด", "must_have": ["ข้าวมันไก่", "ไก่ทอด", "ข้าว"], "optional": ["แตงกวา"]},
    {"menu": "ข้าวมันไก่ทอดไก่ต้ม", "must_have": ["ข้าวมันไก่", "ไก่ทอด", "ไก่ต้ม", "ข้าว"], "optional": ["เลือดไก่ต้ม", "แตงกวา"]},
    {"menu": "ข้าวหมูแดงหมูกรอบ", "must_have": ["ข้าวหมูแดงหมูกรอบ", "ข้าว", "กุนเชียง", "หมูกรอบ", "ไข่ต้ม"], "optional": ["แตงกวา"]},
    {"menu": "ก๋วยเตี๋ยวไก่น่องตุ๋นยาจีน", "must_have": ["ก๋วยเตี๋ยว", "น่องไก่"], "optional": ["เลือดไก่ต้ม", "ไชเท้า"]},
    {"menu": "ก๋วยเตี๋ยวไก่ฉีกตุ๋นยาจีน", "must_have": ["ก๋วยเตี๋ยว", "ไก่ฉีก"], "optional": ["เลือดไก่ต้ม", "ไชเท้า"]},
    {"menu": "ข้าวกะเพราหมูสับ", "must_have": ["กะเพรา", "หมูสับ", "ข้าว"], "optional": []},
    {"menu": "ข้าวกะเพราหมูสับเต้าหู้", "must_have": ["กะเพรา", "หมูสับ", "ข้าว", "เต้าหู้ทอด"], "optional": []},
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

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names.get(cls, f"class_{cls}")
            thai_name = CLASS_TRANSLATIONS.get(class_name, class_name)

            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            draw.text((x1, y1 - 10), f"{thai_name} {conf:.2f}", fill="lime")

            detections.append({
                "class_name": thai_name,
                "confidence": conf
            })
    else:
        print("⚠️ ไม่มีข้อมูลการตรวจจับ")

    # ✅ ตรวจว่าตรงกับเมนูไหน
    detected_names = [d["class_name"] for d in detections]
    matched_menu = None
    matched_components = []

    for rule in MENU_RULES:
        if all(item in detected_names for item in rule["must_have"]):
            matched_menu = rule["menu"]
            matched_components = rule["must_have"] + [x for x in rule["optional"] if x in detected_names]
            break

    # 🔄 แปลงภาพเป็น Base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode()

    # ✅ จัดรูปแบบผลลัพธ์
    if matched_menu:
        components_info = []
        for comp in matched_components:
            conf = next((d["confidence"] for d in detections if d["class_name"] == comp), None)
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
            "detections": detections
        }
