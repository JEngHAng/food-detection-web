เมื่อย้ายเครื่องทำเครื่องใหม่แนะนำให้ทำตามนี้เลย

Python Environment

    - สร้าง : python -m venv .venv

    - เปิดใช้งาน : .\.venv\Scripts\activate

คำสั่ง pip install YOLO + FastAPI

    - pip install -r requirements.txt รันทุกครั้งเมื่อย้ายเครื่อง

ติดตั้ง ngrok

    - ไปที่ https://ngrok.com/download

    - เปิด cmd หรือ PowerShell แล้วพิมพ์: ngrok authtoken <ใส่โทเคน>

รัน FastAPI + ngrok พร้อมกัน

    - เริ่มรันเซิร์ฟเวอร์ FastAPI ก่อน

        - uvicorn server:app --reload --port 8000

    - เปิด PowerShell 

        - ngrok http 8000