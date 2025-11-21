import cv2
import base64
import numpy as np
from fer import FER
from flask import Flask, request, jsonify

app = Flask(__name__)

detector = FER()

@app.route("/emotion", methods=["POST"])
def emotion():
    data = request.json

    if "image" not in data:
        return jsonify({"error": "Image not provided"}), 400

    try:
        img_data = base64.b64decode(data["image"])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except:
        return jsonify({"error": "Invalid image"}), 400

    result = detector.detect_emotions(frame)

    if not result:
        return jsonify({"emotion": "neutral"})

    emotions = result[0]["emotions"]
    top_emotion = max(emotions, key=emotions.get)

    return jsonify({"emotion": top_emotion})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
