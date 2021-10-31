from pandas import read_csv

from models_dict import possible_inputs, possible_outputs
from root_locus_nn import RootLocusNN

def main():
    data = read_csv("data/data.csv")
    for loss in ["mean_squared_error", "mean_squared_logarithmic_error", "mean_absolute_error"]:
        for input in possible_inputs:
            for output in possible_outputs:
                model = RootLocusNN(
                    model_name=input["name"]+"--"+output["name"],
                    data=data, 
                    input_keys=input["keys"],
                    output_keys=output["keys"],
                    scale=True,
                    loss=loss
                )
                model.fit_predict_plot(batch_size=32, epochs=35)
            




if __name__ == "__main__":
    main()