from pandas import read_csv, to_numeric
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

def model_0(input_shape):
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

def plot_metrics(history, metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color="blue", label=metric_name)
    plt.plot(history.history["val_"+metric_name], color="green", label="val_"+metric_name)
    plt.show()

def plot_predictions(test_Y, Y_pred, history, output_heads):
    # for i in range(len(output_heads)):
	#     plot_diff(test_Y[i], Y_pred[i], title=output_heads[i])
    # return
	# Plot RMSE
    for head in output_heads:
	    plot_metrics(history, metric_name="root_mean_squared_error", title=head+" RMSE", ylim=15)

	# Plot loss
    for head in output_heads:
	    plot_metrics(history, metric_name="loss", title=head+" LOSS", ylim=50)
	
def evaluate_model(model, norm_val_x, val_y, output_heads):
    # Test the model and print loss and rmse for both outputs
    losses_and_rmses = model.evaluate(x=norm_val_x, y=val_y)

    print(f"\nloss: {losses_and_rmses[0]}")
    i = 1
    for i in range(len(output_heads)):
        print(f"{output_heads[i]} loss: {losses_and_rmses[i+1]}")
    
    for i in range(len(output_heads)):
        print(f"{output_heads[i]} rmse: {losses_and_rmses[i+len(output_heads)]}")

def format_output(data, output_heads):
    y_list = []
    for column in output_heads:
        y_list.append(np.array(data.pop(column)))
    return y_list


def main():
    data = read_csv("data.csv")
    output_heads = ["pd"]

    unused_indexes = list(range(24))
    unused_indexes.remove(3)  # uclp 0
    unused_indexes.remove(4)  # uclp 1
    unused_indexes.remove(5)  # uclp 2
    unused_indexes.remove(11)  # pd
    data = data.drop(data.columns[unused_indexes], axis=1)
    data = data.astype(complex)

    train, test = train_test_split(data, test_size=0.2, random_state=1)
    train, val = train_test_split(train, test_size=0.2, random_state=1)

    # Get the outputs and format them as np arrays
    y_train = format_output(train, output_heads)
    y_test = format_output(test, output_heads)
    y_val = format_output(val, output_heads)

    # Normalize the training and test data
    x_train_normalized = train
    x_test_normalized = test
    x_val_normalized = val 

    model = model_0(input_shape=(len(train.columns),))

    # Train the model for 200 epochs
    history = model.fit(x_train_normalized, y_train, epochs=1000, batch_size=10, validation_data=(x_test_normalized, y_test))

    evaluate_model(model, x_val_normalized, y_val, output_heads)
    
    # Save model
    model.save("./model_0/", save_format="tf")

    # Restore model
    loaded_model = tf.keras.models.load_model("./model_0/")

    # Run predict
    y_predictions = loaded_model.predict(x_test_normalized)
    
    plot_predictions(y_test, y_predictions, history, output_heads)

if __name__ == "__main__":
    main()