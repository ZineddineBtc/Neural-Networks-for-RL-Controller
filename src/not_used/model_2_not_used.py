"""
inputs: 3 uncompensated closed-loop poles
outputs: Gc-z Gc-p Gc-k
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

from model_parent import ModelParent

class Model_2(ModelParent):

    # def __init__(self, model_name, data, input_keys, output_keys):
    #     super().__init__(model_name, data, input_keys, output_keys)
    #     self.data = self.data.astype(complex)

    def define(self, input_shape):

        input_layer = Input(shape=input_shape)

        first_dense = Dense(units="128", activation="relu")(input_layer)
        y1_output = Dense(units="1", name="z_output")(first_dense)
        
        second_dense = Dense(units="128", activation="relu")(input_layer)
        y2_output = Dense(units="1", name="p_output")(second_dense)
        
        third_dense = Dense(units="128", activation="relu")(input_layer)
        y3_output = Dense(units="1", name="k_output")(third_dense)

        model = Model(inputs=input_layer, outputs=[y1_output, y2_output, y3_output])

        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001),
                    loss={"z_output": "mse", "p_output": "mse", "k_output": "mse"},
                    metrics={
                        "z_output": tf.keras.metrics.RootMeanSquaredError(),
                        "p_output": tf.keras.metrics.RootMeanSquaredError(),
                        "k_output": tf.keras.metrics.RootMeanSquaredError()
                    })
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
        ...
        # super().plot_predictions("root_mean_squared_error", "loss", Y_pred, history)
    
