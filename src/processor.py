import io
import json
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import cv2


class ProcessorFood():
    def __init__(self) -> None:
        self.efficent = "https://tfhub.dev/google/efficientnet/b0/classification/1"
        self.modelPath = 'data/food_recognition_efficientnet_145class.h5'
        self.model = tf.keras.models.load_model(
            self.modelPath, custom_objects={'KerasLayer': hub.KerasLayer}
        )
        self.model.build((None, 250, 250, 3))

        with open('data/class_map.json', 'r', encoding='utf-8') as file:
            self.class_map = json.load(file)
        self.class_map = {v: k for k, v in self.class_map.items()}

    def parallelProcessImage(self, images):
        try:
            with ThreadPoolExecutor() as executor:
                processed = list(executor.map(self.preProcessImage, images))
            return processed
        except Exception as e:
            print(str(e))
            return str(e)

    def preProcessImage(self, images):
        try:
            if isinstance(images, str):
                with open(images, 'rb') as f:
                    image_data = f.read()
                nparr = np.frombuffer(image_data, np.uint8)
                preImage = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                nparr = np.frombuffer(images, np.uint8)
                preImage = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            resizedImage = cv2.resize(preImage, (250, 250))
            preImage = np.array(resizedImage, dtype=np.float32) / 255.0

            if preImage is None:
                return None

            return np.expand_dims(preImage, axis=0)
        except Exception as e:
            return str(e)

    def postProcessImage(self, predicted):
        try:
            key = np.argmax(predicted)
            if key in self.class_map:
                decoded = str(self.class_map[key])

                return decoded
            else:
                print(f"Key {key} not found in class_map")
                return "Key not found in class_map"
        except Exception as e:
            return str(e)

    def predictImage(self, imageData):
        processed = self.preProcessImage(imageData)
        predicted = self.model.predict(processed)
        return self.postProcessImage(predicted)
