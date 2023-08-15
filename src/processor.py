import io
import json

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
from PIL import Image

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

    def preProcessImage(self, images):
        preImage = Image.open(io.BytesIO(images)).resize((250, 250))
        preImage = np.array(preImage) / 255.0

        return np.expand_dims(preImage, axis=0)
    
    def postProcessImage(self, predicted):
        return self.class_map[np.argmax(predicted)]
    
    def predictImage(self, fileName):
        processed = self.preProcessImage(fileName)
        predicted = self.model.predict(processed)
        return self.postProcessImage(predicted)