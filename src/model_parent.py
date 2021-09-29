import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ModelParent:
    def __init__(self, model_name, data, input_keys, output_keys):
        self.model_name = model_name
        self.data = data
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.drop_unused_columns()
        self.set_splits()

    def drop_unused_columns(self):
        keys_to_drop = list(self.data.keys())
        used_keys = self.input_keys + self.output_keys
        for col in used_keys:
            keys_to_drop.remove(col)
        self.data = self.data.drop(keys_to_drop, axis=1)
        self.data = self.data.astype(complex)

    def set_splits(self):
        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=1)
        self.train, self.val = train_test_split(self.train, test_size=0.2, random_state=1)
        self.y_train = self.format_output(self.train)
        self.y_test = self.format_output(self.test)
        self.y_val = self.format_output(self.val)
    
    def format_output(self, data):
        y_list = []
        for column in self.output_keys:
            y_list.append(np.array(data.pop(column)))
        return y_list

    def plot_metrics(self, history, metric_name, title, ylim=5):
        plt.title(title)
        plt.ylim(0, ylim)
        plt.plot(history.history[metric_name], color="blue", label=metric_name)
        plt.plot(history.history["val_"+metric_name], color="green", label="val_"+metric_name)
        plt.show()

    def plot_predictions(self, loss_metric_name, mse_metric_name, Y_pred, history):
        # for i in range(len(output_keys)):
        #     plot_diff(test_Y[i], Y_pred[i], title=output_keys[i])
        # return
        # Plot RMSE
        for head in self.output_keys:
            self.plot_metrics(history, metric_name=mse_metric_name, title=head+" RMSE", ylim=15)

        # Plot loss
        for head in self.output_keys:
            self.plot_metrics(history, metric_name=loss_metric_name, title=head+" LOSS", ylim=50)

    def evaluate_model(self):
        # Test the model and print loss and rmse for both outputs
        losses_and_rmses = self.model.evaluate(x=self.val, y=self.y_val)

        print(f"\nloss: {losses_and_rmses[0]}")
        i = 1
        for i in range(len(self.output_keys)):
            print(f"{self.output_keys[i]} loss: {losses_and_rmses[i+1]}")
        
        for i in range(len(self.output_keys)):
            print(f"{self.output_keys[i]} rmse: {losses_and_rmses[i+len(self.output_keys)]}")
    
    def train_predict_plot(self, epochs):
        self.model = self.define(input_shape=(len(self.train.columns),))
        # Train the model for 200 epochs
        history = self.model.fit(self.train, self.y_train, epochs=epochs, batch_size=10, validation_data=(self.test, self.y_test))
        self.evaluate_model()
        # Save model
        self.model.save("./models/"+self.model_name+"/", save_format="tf")
        # Restore model
        # loaded_model = tf.keras.models.load_model("./model_0/")
        # # Run predict
        y_predictions = self.model.predict(self.test)
        self.plot_predictions(y_predictions, history)
    