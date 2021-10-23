

import scipy.stats as st
import matplotlib.pyplot as plt
from pandas import read_csv
from pathlib import Path

from models_dict import possible_inputs, possible_outputs
from root_locus_nn import RootLocusNN

# def plot(subplotShape, data, keys, outputName):
#     fig, axs = plt.subplots(subplotShape[0], subplotShape[1])
#     for i in range(subplotShape[0]):
#         for j in range(subplotShape[1]):
#             axs[i][j].hist(data[keys[i+j]], bins=10)
#             axs[i][j].set_title(keys[i+j])
#             if i+1!=subplotShape[0]:
#                 axs[i][j].get_xaxis().set_ticks([])

#     plt.savefig("./general/distribution_"+outputName)

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
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

def distribution_experiment(data):
    input_keys = []
    for input in possible_inputs:
        input_keys += input["keys"]
    input_keys = list(set(input_keys))
    input_keys.sort()
    for key in input_keys:
        plot(title="Distribution: "+key,
            column=data[key], 
            bins=25,
            outputName="input_"+key.replace(" ", ""), 
            outputFolder="distribution/inputs/")
        get_best_distribution(data[key])
    
    output_keys = []
    for output in possible_outputs:
        output_keys += output["keys"]
    output_keys = list(set(output_keys))
    output_keys.sort()
    for key in output_keys:
        plot(title="Distribution: "+key,
            column=data[key], 
            bins=25,
            outputName="output_"+key.replace(" ", ""), 
            outputFolder="distribution/outputs/")
        get_best_distribution(data[key])

def epochs_experiment(data):
    epoch_list = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    for input in possible_inputs:
        for output in possible_outputs:
            model_name = input["name"]+"--"+output["name"]
            nn = RootLocusNN(
                model_name=model_name,
                data=data, 
                input_keys=input["keys"],
                output_keys=output["keys"],
                scale=True)
            nn.define_model()
            
            for epoch in epoch_list:
                epoch_folder = "./experiments/epochs/"+model_name+"/"+str(epoch)
                Path(epoch_folder).mkdir(parents=True, exist_ok=True)
                history = nn.model.fit(nn.x_train, nn.y_train, validation_data=(nn.x_test,nn.y_test), batch_size=32, epochs=epoch)
                for key in ["loss", "accuracy"]:
                    plt.title(model_name +": "+ key +" ("+str(epoch)+" epochs)")
                    plt.plot(history.history[key], label=key)
                    plt.plot(history.history["val_"+key], label="validation "+key)
                    plt.legend()
                    plt.savefig(epoch_folder+"/"+key)
                    plt.close()

            
def main():
    data = read_csv("data/data.csv")
    # distribution_experiment(data, possible_inputs, possible_outputs)
    epochs_experiment(data)


if __name__=="__main__":
    main()