from flask import Flask, request, jsonify
from werkzeug.datastructures import FileStorage
from src.processor import *

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files.get('image')
    
    if image_file and isinstance(image_file, FileStorage):
        # 이미지 파일을 임시 경로에 저장합니다.
        image_path = "/tmp/temp_image.jpg"
        image_file.save(image_path)

        # ProcessorFood 클래스의 predict 메서드에 이미지 경로를 전달합니다.
        prediction = ProcessorFood.predict(image_path)
        return jsonify({"result": prediction})
    else:
        return jsonify({"error": "Image file is missing or invalid."}), 400

if __name__ == '__main__':
    app.run()
