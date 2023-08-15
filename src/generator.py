import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator

generated = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
data_dir = 'data/images'

trained = generated.flow_from_directory(
    data_dir,
    target_size = (250, 250),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'training'
)
validated = generated.flow_from_directory(
    data_dir,
    target_size = (250, 250),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation'
)

efficentData = "https://tfhub.dev/google/efficientnet/b0/classification/1"
model = tf.keras.Sequential([
    hub.KerasLayer(efficentData, trainable = False, input_shape = (250,250,3), name = 'Resnet_V2_50'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(145, activation = 'softmax', name = 'outputLayer')
])
model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = ['accuracy']
)
result = model.fit(
    trained,
    epochs = 50,
    validation_data = validated
)

model_name = 'efficientnet_145class.h5'
model.save(model_name, save_format='h5')
