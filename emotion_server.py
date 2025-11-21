import cv2
import numpy as np
from fer import FER
from flask import Flask, request, jsonify

app = Flask(__name__)
detector = FER()

@app.route("/emotion", methods=["POST"])
def emotion():
    # نتأكد إن فيه ملف صورة اسمه "image" في الطلب
    if "image" not in request.files:
        return jsonify({"error": "no image file with key 'image'"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "could not decode image"}), 400

    # تحليل المشاعر
    result = detector.detect_emotions(img)

    if not result:
        return jsonify({"emotion": "neutral"})

    emotions = result[0]["emotions"]
    top_emotion = max(emotions, key=emotions.get)

    return jsonify({"emotion": top_emotion})

@app.route("/", methods=["GET"])
def index():
    return "EmoPlay emotion server is running!"

if __name__ == "__main__":
    # للتجربة محليًا فقط
    app.run(host="0.0.0.0", port=5000)
