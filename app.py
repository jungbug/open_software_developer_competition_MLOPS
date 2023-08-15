import os

from flask import Flask, request, jsonify
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from src.processor import *
from src.matrix_cluster import *

app = Flask(__name__)

@app.route('/predict/image', methods=['POST'])
def predict():
    try:
        image_file = request.files.get('image')
        if image_file and isinstance(image_file, FileStorage) and image_file.filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join("/tmp", secure_filename("temp_image.jpg"))
            image_file.save(image_path)

            prediction = ProcessorFood.predict(image_path)
            return jsonify({"result": prediction})
        else:
            return jsonify({"error": "Image file is missing or invalid."}), 400
    except Exception:
        return jsonify({"error": "An internal error occurred."}), 500
    
@app.route('/predict/video', methods=['POST'])
def predict_video():
    try:
        video_file = request.files.get('video')
        if video_file and isinstance(video_file, FileStorage) and video_file.filename.endswith(('.mp4', '.avi', '.flv', '.mkv')):
            video_path = os.path.join("/tmp", secure_filename("temp_video.mp4"))
            video_file.save(video_path)

            prediction = video_parallel(video_path)
            return jsonify({"result": prediction})
        else:
            return jsonify({"error": "Video file is missing or invalid."}), 400
    except Exception as e:
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    app.run()
