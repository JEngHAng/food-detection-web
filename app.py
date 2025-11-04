from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO("best.pt")  # ใช้โมเดลที่เทรนแล้ว

# 🔸 ใส่ URL ของกล้องมือถือที่ได้จาก IP Webcam
camera_url = "http://192.168.1.12:8080/video"  # <== เปลี่ยนตรงนี้ให้เป็นของคุณ

def gen_frames():
    cap = cv2.VideoCapture(camera_url)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            annotated = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
