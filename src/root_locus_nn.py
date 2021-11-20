import pandas
from pathlib import Path
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial as poly
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# from control_theory import required_phase, compensator_zpk, closed_loop_tf, parameters_switch

class RootLocusNN:
    def __init__(self, data, scale, hiddenLayers_count, loss="mean_squared_error"):
        self.input_keys = ["PO max", "Ts max", "uoln", "uold 0", "uold 1", "uold 2"]
        self.output_keys = ["Gc-z", "Gc-p", "Gc-k"]  
        self.data = data
        self.set_data(data, scale)
        self.hiddenLayers_count = hiddenLayers_count-1  # output-layer: Dense() => hidden-layer
        if self.hiddenLayers_count < 0:
            self.hiddenLayers_count = 0
        self.loss = loss    
            
    def set_data(self, data, scale):
        self.x = data[self.input_keys]
        self.x = self.x.astype(float)
        print("x info: ")
        print(self.x.info())
        self.y = data[self.output_keys]
        self.y = self.y.astype(float)
        print("y info: ")
        print(self.y.info())
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=0)
        if scale:
            self.x_train = self.normalize(self.x_train)
            self.x_test = self.normalize(self.x_test)

    def normalize(self, df):
        arr = df.values
        min_max_scaler = MinMaxScaler()
        arr_scaled = min_max_scaler.fit_transform(arr)
        return pandas.DataFrame(arr_scaled)
        
    def define_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_dim=len(self.input_keys)))
        for i in range(self.hiddenLayers_count-1):
            self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(len(self.output_keys), activation='linear'))
        self.model.compile(optimizer='adam', loss=self.loss)
        self.model.summary()
        
    def fit_predict_plot(self, batch_size, epochs):
        self.define_model()
        self.history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test,self.y_test), batch_size=batch_size, epochs=epochs)
        folder_path = "./main_results/"
        Path(folder_path+"/model").mkdir(parents=True, exist_ok=True)
        Path(folder_path+"/plot").mkdir(parents=True, exist_ok=True)
        self.model.save(folder_path+"/model")
        self.y_pred = self.model.predict(self.x_test)
        self.rearrange_outputs(self.output_keys)
        self.plot_loss(folder_path+"/plot")
        self.plot_predictions(folder_path+"/plot", self.output_keys)
        self.calculate_prediction_distortion()
    
    def rearrange_outputs(self, titles):
        self.y_test = self.y_test.to_numpy()  
        self.y_rearrenged = []
        for i in titles:
            self.y_rearrenged.append([[], []])
        for outputs in self.y_test:
            for i in range(len(self.output_keys)):
                self.y_rearrenged[i][0].append(outputs[i])
        for outputs in self.y_pred:
            for i in range(len(self.output_keys)):
                self.y_rearrenged[i][1].append(outputs[i])

    def plot_loss(self, folder_path):
        plt.title("Loss")
        plt.plot(self.history.history["loss"], label="loss")
        plt.plot(self.history.history["val_loss"], label="validation loss")
        plt.legend()
        plt.title(self.loss)
        plt.savefig(folder_path+"/loss")
        plt.close()
        
    def plot_predictions(self, folder_path, titles):      
        fig, axs = plt.subplots(len(titles))
        for i in range(len(titles)):
            axs[i].plot(self.y_rearrenged[i][0], label="expected")
            axs[i].plot(self.y_rearrenged[i][1], label="predicted")
            axs[i].set_title("Prediction: "+titles[i])
            axs[i].legend()
            if i+1!=len(titles):
                axs[i].get_xaxis().set_ticks([])
        plt.suptitle(self.loss)
        plt.tight_layout()
        plt.savefig(folder_path+"/predictions")
        plt.close()

    def calculate_prediction_distortion(self):
        print(self.loss)
        for i in range(len(self.output_keys)):
            total_difference = 0
            for j in range(len(self.y_rearrenged[0])):
                total_difference += abs(self.y_rearrenged[i][j][0]-self.y_rearrenged[i][j][1])
            print("total difference ("+self.output_keys[i]+"): "+str(total_difference))

    # def check_specifications(self):      
    #     for i in range(len(self.y_pred)):
    #         predicted_output, uoln, uolds, uolps, uolz, PO_max, Ts_max = self.get_vars(i)
    #         if "pd -r" in self.output_keys:
    #             pd = complex(predicted_output[0], predicted_output[1])
    #             required_additional_phase = required_phase(pd, uolz, uolps)
    #             zc, pc, kc = compensator_zpk("complex", pd, uolps, required_additional_phase, uoln, uolds)
    #         else:
    #             zc, pc, kc = predicted_output
                
            # tf_nominator, tf_denominator = closed_loop_tf("complex", zc, kc, pc, uoln, uolps)
            # closed_loop_poles = tf_denominator.roots()
            # print(zc, pc, kc, closed_loop_poles)
            # c, PO, Ts = parameters_switch("distinct", closed_loop_poles)
            
    # def get_vars(self, i):
    #     predicted_output = self.y_pred[i]
    #     uoln = poly([self.data["uoln"][i]])
    #     uolds = poly([self.data["uold 0"][i], self.data["uold 1"][i], self.data["uold 2"][i], self.data["uold 3"][i]])
    #     uolps = [self.data["uolp 0"][i], self.data["uolp 1"][i], self.data["uolp 2"][i]]
    #     uolz = []
    #     PO_max = self.data["PO max"][i]
    #     Ts_max = self.data["Ts max"][i]
    #     return predicted_output, uoln, uolds, uolps, uolz, PO_max, Ts_max
            
