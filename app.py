from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)

model = YOLO("yolov8n.pt")

@app.route('/')
def home():
    return "YOLO Server Running"

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']

    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    results = model(frame)

    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detected_objects.append(label)

    print("Detected:", detected_objects)

    return jsonify({"objects": detected_objects})

app.run(host='0.0.0.0', port=5000)