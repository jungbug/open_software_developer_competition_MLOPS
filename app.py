import os
import json
import tempfile

from flask import Flask, request, jsonify, Response
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from src.processor import *
from src.pose_processor import *

app = Flask(__name__)


@app.route('/predict/image', methods=['POST'])
def predict():
    try:
        image_file = request.files.get('image')
        processor = ProcessorFood()
        if image_file and isinstance(image_file, FileStorage) and image_file.filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(
                "/tmp", secure_filename("temp_image.jpg"))
            image_file.save(image_path)

            with open(image_path, 'rb') as f:
                image_data = f.read()

            prediction = processor.predictImage(image_data)
            response = json.dumps({"result": prediction}, ensure_ascii=False)
            return Response(response, content_type="application/json; charset=utf-8")
        else:
            return jsonify({"error": "Image file is missing or invalid."}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": "An internal error occurred."}), 500


@app.route('/predict/video', methods=['POST'])
def predict_video():
    try:
        video_file = request.files.get('video')
        processor = ProcessorPose()
        if video_file and isinstance(video_file, FileStorage) and video_file.filename.endswith(('.mp4', '.avi', '.flv', '.mkv')):
            tempDir = tempfile.gettempdir()
            video_path = os.path.join(
                tempDir, secure_filename("temp_video.mp4"))
            video_file.save(video_path)

            prediction = processor.predictImage(video_path)
            response = json.dumps({"result": prediction}, ensure_ascii=False)
            return Response(response, content_type="application/json; charset=utf-8")
        else:
            return jsonify({"error": "Video file is missing or invalid."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()
