import json
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import cv2


class ProcessorPose():
    def __init__(self) -> None:
        self.efficent = "https://tfhub.dev/google/efficientnet/b0/classification/1"
        self.modelPath = 'data/pose.h5'
        self.model = tf.keras.models.load_model(
            self.modelPath, custom_objects={'KerasLayer': hub.KerasLayer}
        )
        self.model.build((None, 250, 250, 3))

        with open('data/health_map.json', 'r', encoding='utf-8') as file:
            self.health_map = json.load(file)
        self.health_map = {v: k for k, v in self.health_map.items()}

    def parallelProcessImage(self, images):
        try:
            with ThreadPoolExecutor() as executor:
                processed = list(executor.map(self.preProcessImage, images))
            return processed
        except Exception as e:
            print(str(e) + " in parallelProcessImage")
            return str(e)

    def preProcessImage(self, images):
        try:
            print(f"Type of images: {type(images)}")

            if isinstance(images, str):
                with open(images, 'rb') as f:
                    image_data = f.read()

                nparr = np.frombuffer(image_data, np.uint8)
                preImage = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                nparr = np.frombuffer(images, np.uint8)
                preImage = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if preImage is None:
                return None

            resizedImage = cv2.resize(preImage, dsize=(
                400, 400), interpolation=cv2.INTER_CUBIC)
            return np.expand_dims(resizedImage, axis=0)
        except Exception as e:
            print(str(e) + " in preProcessImage")
            return str(e)

    def postProcessImage(self, predicted):
        try:
            key = np.argmax(predicted)
            if key in self.health_map:
                decoded = str(self.health_map[key])

                return decoded
            else:
                print(f"Key {key} not found in class_map")
                return "Key not found in class_map"
        except Exception as e:
            print(str(e) + " in postProcessImage")
            return str(e)

    def predictImage(self, media):
        vid = cv2.VideoCapture(media)
        resultArr = [0 for _ in range(len(self.health_map))]

        while vid.isOpened():
            ret, image = vid.read()
            if ret == False:
                break
            if image is None:
                continue
            if int(vid.get(1)) % 60 == 0:
                processed = self.preProcessImage(image)
                if processed is None:
                    continue
                predicted = self.model.predict(processed)

                result = self.postProcessImage(predicted)
                for i, idx in enumerate(self.health_map):
                    if result == idx:
                        resultArr[i] += 1

        return self.health_map[resultArr.index(max(resultArr))]
