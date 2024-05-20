import tensorflow as tf
from keras.datasets import mnist
def load_data():
    (x_train, y_train),(x_test,y_test)= mnist.load_data()
    x_train,x_test = x_train/255.0,x_test/255.0
    return (x_train,y_train),(x_test,y_test)
