import mlflow
import mlflow.tensorflow
from data import load_data
from model import create_model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
   
    mlflow.start_run()
   
    model.fit(x_train, y_train, epochs=5)
    model.save('src/models/model.h5')
   
    mlflow.log_param("epochs", 5)
    mlflow.tensorflow.log_model(model, "model")
   
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    mlflow.log_metric("test_accuracy", test_acc)
   
    mlflow.end_run()

if __name__ == '__main__':
    train_model()