"""
inputs: 3 uncompensated closed-loop poles
outputs: pd
"""

from pandas import read_csv, to_numeric
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

from model_parent import ModelParent

class Model_0(ModelParent):
    def __init__(self, input_shape, output_heads):
        super(output_heads, input_shape)
    
    def define_model(self):
        input_layer = Input(shape=self.input_shape)
        dense = Dense(units="128", activation="relu")(input_layer)
        y_output = Dense(units="1", name="pd_output")(dense)
        model = Model(inputs=input_layer, outputs=[y_output])
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.001),
            loss={"pd_output": "mse"},
            metrics={"pd_output": tf.keras.metrics.RootMeanSquaredError()})
        return model
    
