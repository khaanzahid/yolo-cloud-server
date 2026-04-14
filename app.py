from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)

# 🔥 Lazy loading (prevents 502 crash)
model = None

# =========================
# HOME ROUTE
# =========================
@app.route('/')
def home():
    return "YOLO Server Running"

# =========================
# DETECTION ROUTE
# =========================
@app.route('/detect', methods=['POST'])
def detect():
    global model

    try:
        print("---- New Request ----")

        # 🔥 Load model only when needed
        if model is None:
            print("Loading YOLO model...")
            model = YOLO("yolov8n.pt")

        # 📸 Get raw image from ESP32
        img_bytes = request.data

        if len(img_bytes) == 0:
            print("Empty image received!")
            return "Empty Image", 400

        print("Image size:", len(img_bytes))

        # 🔄 Convert to OpenCV image
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("Image decode failed!")
            return "Decode Failed", 400

        print("Image decoded successfully")

        # 🧠 YOLO Detection
        results = model(frame)

        detected_objects = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                detected_objects.append(label)

        print("Detected:", detected_objects)

        return jsonify({"objects": detected_objects})

    except Exception as e:
        print("🔥 FULL ERROR:", str(e))
        return "Server Error", 500

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
