import scipy.stats as st
import matplotlib.pyplot as plt
from pandas import read_csv
from pathlib import Path

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
        test_split=0.2,
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

def hiddenLayers_experiment(data):
    hiddenLayers_options = [1, 2, 3, 4, 5, 6]
    for count in hiddenLayers_options:
        Path("./experiments/hidden-layers/count-"+str(count)).mkdir(parents=True, exist_ok=True)
        RootLocusNN(
            data=data, 
            test_split=0.2,
            scale=True,
            hiddenLayers_count=count,
            loss="mse").fit_predict_plot(
                output_folder="./experiments/hidden-layers/count-"+str(count)+"/",
                toSave=False, batch_size=32, epochs=35, predictions_title=str(count))

def loss_experiment(data):
    losses = ["mse", "msle", "mae"]
    epochs = 35
    hiddenLayers_count = 2
    for loss in losses:
        Path("./experiments/loss/"+loss).mkdir(parents=True, exist_ok=True)
        RootLocusNN(
            data=data, 
            test_split=0.2,
            scale=True,
            hiddenLayers_count=hiddenLayers_count,
            loss=loss).fit_predict_plot(
                output_folder="./experiments/loss/"+loss+"/",
                toSave=False, batch_size=32, epochs=epochs, predictions_title=loss)

def testSplit_experiment(data):
    loss = "mse"
    epochs = 35
    hiddenLayers_count = 2
    splits = [0.1, 0.3, 0.5, 0.7]
    for split in splits:
        folder = "./experiments/test-split/split_"+str(split)
        Path(folder).mkdir(parents=True, exist_ok=True)
        RootLocusNN(
            data=data, 
            test_split=split,
            scale=True,
            hiddenLayers_count=hiddenLayers_count,
            loss=loss).fit_predict_plot(
                output_folder=folder+"/", toSave=False, 
                batch_size=32, epochs=epochs, predictions_title=str(split))

def main():
    data = read_csv("data/data.csv")
    distribution_experiment(data)
    epoch_experiment(data)
    hiddenLayers_experiment(data)
    loss_experiment(data)
    testSplit_experiment(data)

if __name__=="__main__":
    main()