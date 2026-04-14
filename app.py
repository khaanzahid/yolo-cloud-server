from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)

# 🔥 Lazy load model (fixes 502 crash)
model = None

# Home route
@app.route('/')
def home():
    return "YOLO Server Running"

# Detection route
@app.route('/detect', methods=['POST'])
def detect():
    global model

    try:
        # 🔥 Load model only when needed
        if model is None:
            print("Loading YOLO model...")
            model = YOLO("yolov8n.pt")

        # 📸 Get raw image from ESP32
        img_bytes = request.data

        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return "Invalid Image", 400

        # 🧠 Run YOLO
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
        print("Error:", e)
        return "Server Error", 500

# Run server
app.run(host='0.0.0.0', port=5000)
