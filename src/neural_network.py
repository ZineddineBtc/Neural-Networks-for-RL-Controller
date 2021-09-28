import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.utils import Bunch

def load_my_dataset(filepath):
    with open(filepath) as csv_file:
        data_file = csv.reader(csv_file)
        n_samples = 150  # number of data rows, don't count header
        n_features = 4  # number of columns for features, don't count target column
        feature_names = ['f1','f2','f3','f4'] 
        target_names = ['t1','t2','t3']
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1], dtype=np.float64)
            target[i] = np.asarray(sample[-1], dtype=np.int)

    return Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)

def get_data():
	# Loading the Boston Housing dataset
	boston = load_boston()
	print(boston.target)
	return
	# Initializing the dataframe
	data = pd.DataFrame(boston.data)

	# Adding the feature names to the dataframe
	data.columns = boston.feature_names
	print(data.columns)

	# Adding target variable to dataframe
	data["PRICE"] = boston.target

	print(data.head())

	# Check the shape of dataframe
	print(data.shape)
	print(data.columns)

	print(type(data))
	return data

def norm(x, train_stats):
    return (x - train_stats["mean"]) / train_stats["std"]

def format_output(data):
	y1 = data.pop("PRICE")
	y1 = np.array(y1)
	y2 = data.pop("PTRATIO")
	y2 = np.array(y2)
	y3 = data.pop("LSTAT")
	y3 = np.array(y2)
	return y1, y2, y3

def define_model(input_shape):
	input_layer = Input(shape=input_shape)

	first_dense = Dense(units="128", activation="relu")(input_layer)
	y1_output = Dense(units="1", name="price_output")(first_dense)
	
	second_dense = Dense(units="128", activation="relu")(first_dense)
	y2_output = Dense(units="1", name="ptratio_output")(second_dense)
	
	third_dense = Dense(units="128", activation="relu")(first_dense)
	y3_output = Dense(units="1", name="lstat_output")(third_dense)

	model = Model(inputs=input_layer, outputs=[y1_output, y2_output, y3_output])

	model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001),
				loss={"price_output": "mse", "ptratio_output": "mse", "lstat_output": "mse"},
				metrics={
					"price_output": tf.keras.metrics.RootMeanSquaredError(),
					"ptratio_output": tf.keras.metrics.RootMeanSquaredError(),
					"lstat_output": tf.keras.metrics.RootMeanSquaredError()
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

def plot_metrics(history, metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color="blue", label=metric_name)
    plt.plot(history.history["val_"+metric_name], color="green", label="val_"+metric_name)
    plt.show()

def plot_predictions(test_Y, Y_pred, history):
	plot_diff(test_Y[0], Y_pred[0], title="PRICE")
	plot_diff(test_Y[1], Y_pred[1], title="PTRATIO")
	plot_diff(test_Y[2], Y_pred[2], title="LSTAT")

	# Plot RMSE
	plot_metrics(history, metric_name="price_output_root_mean_squared_error", title="PRICE RMSE", ylim=15)
	plot_metrics(history, metric_name="ptratio_output_root_mean_squared_error", title="PTRATIO RMSE", ylim=7)
	plot_metrics(history, metric_name="lstat_output_root_mean_squared_error", title="LSTAT RMSE", ylim=7)

	# Plot loss
	plot_metrics(history, metric_name="price_output_loss", title="PRICE LOSS", ylim=50)
	plot_metrics(history, metric_name="ptratio_output_loss", title="PTRATIO LOSS", ylim=15)
	plot_metrics(history, metric_name="lstat_output_loss", title="LSTAT LOSS", ylim=15)

def evaluate_model(model, norm_val_x, val_y):
	# Test the model and print loss and rmse for both outputs
	loss, Y1_loss, Y2_loss, Y3_loss, Y1_rmse, Y2_rmse, Y3_rmse = model.evaluate(x=norm_val_x, y=val_y)

	print(f"\nloss: {loss}")
	print(f"price_loss: {Y1_loss}")
	print(f"ptratio_loss: {Y2_loss}")
	print(f"lstat_loss: {Y3_loss}")
	print(f"price_rmse: {Y1_rmse}")
	print(f"ptratio_rmse: {Y2_rmse}")
	print(f"lstat_rmse: {Y3_rmse}\n")

def main():

	data = get_data()


	# Split the data 
	train, test = train_test_split(data, test_size=0.2, random_state = 1)
	train, val = train_test_split(train, test_size=0.2, random_state = 1)

	# Get Y1 and Y2 as the 2 outputs and format them as np arrays
	train_stats = train.describe()
	train_stats.pop("PRICE")
	train_stats.pop("PTRATIO")
	train_stats.pop("LSTAT")
	train_stats = train_stats.transpose()
	y_train = format_output(train)
	y_test = format_output(test)
	y_val = format_output(val)

	# Normalize the training and test data
	x_train_normalized = np.array(norm(train, train_stats))
	x_test_normalized = np.array(norm(test, train_stats))
	x_val_normalized = np.array(norm(val, train_stats))

	model = define_model(input_shape=(len(train.columns),))

	# Train the model for 200 epochs
	history = model.fit(x_train_normalized, y_train, epochs=10, batch_size=10, validation_data=(x_test_normalized, y_test))
	
	evaluate_model(model, x_val_normalized, y_val)

	# Run predict
	y_predictions = model.predict(x_test_normalized)
	plot_predictions(y_test, y_predictions, history)

	# Save model
	model.save("./model_boston/", save_format="tf")

	# Restore model
	loaded_model = tf.keras.models.load_model("./model_boston/")

	# Run predict with restored model
	predictions = loaded_model.predict(x_test_normalized)
	price_predictions = predictions[0]
	ptratio_predictions = predictions[1]
	lstat_predictions = predictions[2]


if __name__ == "__main__":
	main()





