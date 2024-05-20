import tensorflow as tf
from data import load_data

def evaluate_model(model_path='src/models/model.h5'):
    _, (x_test, y_test) = load_data()
    model = tf.keras.models.load_model(model_path)
   
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

if __name__ == '__main__':
    evaluate_model()