import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ModelParent:
    def __init__(self, input_shape, output_heads):
        self.output_heads = output_heads
        self.input_shape = input_shape
    
    def define_model():
        ...

    def evaluate_model(self, model, norm_val_x, val_y):
        # Test the model and print loss and rmse for both outputs
        losses_and_rmses = model.evaluate(x=norm_val_x, y=val_y)

        print(f"\nloss: {losses_and_rmses[0]}")
        i = 1
        for i in range(len(self.output_heads)):
            print(f"{self.output_heads[i]} loss: {losses_and_rmses[i+1]}")
        
        for i in range(len(self.output_heads)):
            print(f"{self.output_heads[i]} rmse: {losses_and_rmses[i+len(self.output_heads)]}")

    def format_output(self, data):
        y_list = []
        for column in self.output_heads:
            y_list.append(np.array(data.pop(column)))
        return y_list
    
    def plot_metrics(history, metric_name, title, ylim=5):
        plt.title(title)
        plt.ylim(0, ylim)
        plt.plot(history.history[metric_name], color="blue", label=metric_name)
        plt.plot(history.history["val_"+metric_name], color="green", label="val_"+metric_name)
        plt.show()
    
    def drop_unused_columns(data, unused_columns):
        unused_indexes = list(range(24))
        for col in unused_columns:
            unused_indexes.remove(col)
        data = data.drop(data.columns[unused_indexes], axis=1)
        data = data.astype(complex)
        return data
    
    def predict(self, data, to_train):
        train, test = train_test_split(data, test_size=0.2, random_state=1)
        train, val = train_test_split(train, test_size=0.2, random_state=1)

        # Get the outputs and format them as np arrays
        y_train = self.format_output(train, self.output_heads)
        y_test = self.format_output(test, self.output_heads)
        y_val = self.format_output(val, self.output_heads)

        if to_train:
            model = self.define_model()

        # Train the model for 200 epochs
        history = model.fit(train, y_train, epochs=1000, batch_size=10, validation_data=(test, y_test))

        self.evaluate_model(model, val, y_val)
        
        if save_model!="":
            # Save model
            model.save("./model_0/", save_format="tf")

        # Run predict
        y_predictions = model.predict(test)
        
        self.plot_predictions(y_test, y_predictions, history)