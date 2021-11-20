import scipy.stats as st
import matplotlib.pyplot as plt
from pandas import read_csv
from pathlib import Path

from models_dict import possible_inputs, possible_outputs
from root_locus_nn import RootLocusNN


def plot(title, column, bins, outputName, outputFolder):
    plt.hist(column, bins=bins)
    plt.title(title)
    plt.savefig("./experiments/"+outputFolder+"distribution_"+outputName)
    plt.close()

def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))
    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value
    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist])+"\n")
    return best_dist, best_p, params[best_dist]

def distribution_experiment(data):
    columns = ["PO max", "Ts max", "uoln", "uold 0", "uold 1", "uold 2", "Gc-z", "Gc-p", "Gc-k"]
    for key in columns:
        plot(title="Distribution: "+key,
            column=data[key], 
            bins=25,
            outputName=key.replace(" ", ""), 
            outputFolder="distribution/")
        print("KEY: "+key)
        get_best_distribution(data[key])

def epoch_experiment(data):
    epoch_list = [10, 20, 50, 100, 200, 500]
    nn = RootLocusNN(
        data=data, 
        scale=True,
        hiddenLayers_count=3,
        loss="mse")
    nn.define_model()
    for epoch in epoch_list:
        epoch_folder = "./experiments/epochs/"
        Path(epoch_folder).mkdir(parents=True, exist_ok=True)
        history = nn.model.fit(nn.x_train, nn.y_train, validation_data=(nn.x_test,nn.y_test), batch_size=32, epochs=epoch)
        plt.title(str(epoch)+" epochs")
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="validation loss")
        plt.legend()
        plt.savefig(epoch_folder+"/"+str(epoch))
        plt.close()

def loss_experiment(data):
    epoch_list = [10, 20, 50, 100, 200, 500]
    losses = ["mean_squared_error", "mean_squared_logarithmic_error", "mean_absolute_error"]
    model_name = "uoln-uclds--Gc"
    for loss in losses:
        nn = RootLocusNN(
            model_name=model_name,
            data=data, 
            input_keys=["uoln", "ucld 0", "ucld 1", "ucld 2", "ucld 3"],
            output_keys=["Gc-z", "Gc-p", "Gc-k"],
            scale=True,
            loss=loss)
        nn.define_model()
                
        for epoch in epoch_list:
            epoch_folder = "./experiments/epochs/"+loss
            Path(epoch_folder).mkdir(parents=True, exist_ok=True)
            history = nn.model.fit(nn.x_train, nn.y_train, validation_data=(nn.x_test,nn.y_test), batch_size=32, epochs=epoch)
            key = "loss"
            plt.title(key +" ("+str(epoch)+" epochs)")
            plt.plot(history.history[key], label=key)
            plt.plot(history.history["val_"+key], label="validation "+key)
            plt.legend()
            plt.savefig(epoch_folder+"/"+key+"_"+str(epoch))
            plt.close()

            
def main():
    data = read_csv("data/data.csv")
    # distribution_experiment(data)
    epoch_experiment(data)

if __name__=="__main__":
    main()