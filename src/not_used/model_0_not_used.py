"""
inputs: 3 uncompensated closed-loop poles
outputs: pd
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

from model_parent import ModelParent

class Model_0(ModelParent):

    # def __init__(self, model_name, data, input_keys, output_keys):
    #     super().__init__(model_name, data, input_keys, output_keys)
    #     self.data = self.data.astype(complex)

    def define(self, input_shape):
        # Input: 3 uncompensated closed-loop poles
        # Output: pd

        input_layer = Input(shape=input_shape)

        dense = Dense(units="128", activation="relu")(input_layer)
        y_output = Dense(units="1", name="pd_output")(dense)
        
        model = Model(inputs=input_layer, outputs=[y_output])
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.001),
            loss={"pd_output": "mse"},
            metrics={"pd_output": tf.keras.metrics.RootMeanSquaredError()})
        return model

    def plot_diff(y_true, y_pred, title=""):
        plt.scatter(y_true, y_pred)
        plt.title(title)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.axis("equal")
        plt.axis("square")
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plt.plot([-100, 100], [-100, 100])
        plt.show()

    def plot_predictions(self, Y_pred, history):
        super().plot_predictions(Y_pred, history)
    
