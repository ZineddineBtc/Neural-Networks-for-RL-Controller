import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

class RootLocusNN:
    def __init__(self, model_name, data, input_keys, output_keys, scale, loss="mean_squared_error"):
        self.model_name = model_name
        self.loss = loss
        self.input_keys = ["PO max", "Ts max"]
        self.input_keys += input_keys
        self.output_keys = output_keys
        self.set_data(data, scale)
    
    def set_data(self, data, scale):
        self.x = data[self.input_keys]
        self.x = self.x.astype(float)
        print("x info: ")
        print(self.x.info())
        self.y = data[self.output_keys]
        self.y = self.y.astype(float)
        print("y info: ")
        print(self.y.info())
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.08, random_state=0)
        if scale:
            self.x_train = self.normalize(self.x_train)
            self.x_test = self.normalize(self.x_test)

    def normalize(self, df):
        arr = df.values
        min_max_scaler = MinMaxScaler()
        arr_scaled = min_max_scaler.fit_transform(arr)
        return pd.DataFrame(arr_scaled)
        
    def define_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_dim=len(self.input_keys)))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(len(self.output_keys), activation='linear'))
        self.model.compile(optimizer='adam', loss=self.loss)
        self.model.summary()
        
    def fit_predict_plot(self, batch_size, epochs):
        self.define_model()
        self.history = self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)
        __model = self.model_name+"_epoch_"+str(epochs)
        folder_path = "./main_results/"+__model
        Path(folder_path+"/model").mkdir(parents=True, exist_ok=True)
        Path(folder_path+"/plot").mkdir(parents=True, exist_ok=True)
        self.model.save(folder_path+"/model")
        self.y_pred = self.model.predict(self.x_test)
        self.plot_loss(folder_path+"/plot")
        self.plot_predictions(folder_path+"/plot", self.output_keys)
    
    def plot_loss(self, folder_path):
        plt.title("Loss")
        plt.plot(self.history.history["loss"], color="blue", label="loss")
        plt.savefig(folder_path+"/loss_"+self.model_name)
        
    def plot_predictions(self, folder_path, titles):
        self.y_test = self.y_test.to_numpy()
        
        y_plot = []
        for i in titles:
            y_plot.append([[], []])

        for outputs in self.y_test:
            for i in range(len(titles)):
                y_plot[i][0].append(outputs[i])
        
        for outputs in self.y_pred:
            for i in range(len(titles)):
                y_plot[i][1].append(outputs[i])
        
        fig, axs = plt.subplots(len(titles))
        for i in range(len(titles)):
            axs[i].plot(y_plot[i][0], label="expected")
            axs[i].plot(y_plot[i][1], label="predicted")
            axs[i].set_title("Prediction: "+titles[i])
            axs[i].legend()
            if i+1!=len(titles):
                axs[i].get_xaxis().set_ticks([])
        plt.savefig(folder_path+"/accuracy_"+self.model_name)
