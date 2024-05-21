import tensorflow as tf
import numpy as np

def load_model(model_path='src/models/model.h5'):
    return tf.keras.models.load_model(model_path)

def prediction(model, data):
    data = np.array(data) / 255.0
    data = data.reshape(1, 28, 28)
    predictions = model.predict(data)
    return np.argmax(predictions[0])