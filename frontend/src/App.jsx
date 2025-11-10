import { useRef, useState } from "react";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [detectedImage, setDetectedImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const BACKEND_URL = "http://127.0.0.1:8000/detect_image";

  // เปิดกล้อง
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    } catch (err) {
      alert("ไม่สามารถเข้าถึงกล้อง: " + err.message);
    }
  };

  // ถ่ายภาพจากกล้องแล้วส่งไป backend
  const captureAndDetect = async () => {
    if (!videoRef.current) return;
    setLoading(true);

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      const form = new FormData();
      form.append("file", blob, "capture.jpg");

      try {
        const response = await fetch(BACKEND_URL, {
          method: "POST",
          body: form,
        });
        const blobResult = await response.blob();
        const imageUrl = URL.createObjectURL(blobResult);
        setDetectedImage(imageUrl);
      } catch (err) {
        alert("Error: " + err.message);
      } finally {
        setLoading(false);
      }
    }, "image/jpeg");
  };

  // อัปโหลดไฟล์จากเครื่อง
  const onFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);

    const form = new FormData();
    form.append("file", file);

    try {
      const response = await fetch(BACKEND_URL, {
        method: "POST",
        body: form,
      });
      const blobResult = await response.blob();
      const imageUrl = URL.createObjectURL(blobResult);
      setDetectedImage(imageUrl);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>🍱 YOLOv8 18-Class Detector</h1>

      <div>
        <button onClick={startCamera}>📷 Start Camera</button>
        <button onClick={captureAndDetect} disabled={loading}>
          🔍 Capture & Detect
        </button>
        <input type="file" accept="image/*" onChange={onFileChange} />
      </div>

      <div style={{ marginTop: 10 }}>
        <video ref={videoRef} style={{ width: 400 }} />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      {loading && <p>⏳ กำลังตรวจจับ...</p>}

      {detectedImage && (
        <div style={{ marginTop: 20 }}>
          <h3>✅ Detection Result</h3>
          <img src={detectedImage} alt="result" width={400} />
        </div>
      )}
    </div>
  );
}
