from pandas import read_csv

from models_dict import possible_inputs, possible_outputs
from root_locus_nn import RootLocusNN

def main():
    data = read_csv("data/data.csv")
    
    epochs = 1

    for input in possible_inputs:
        for output in possible_outputs:
            RootLocusNN(
                model_name=input["name"]+"--"+output["name"],
                data=data, 
                input_keys=input["keys"],
                output_keys=output["keys"],
                scale=True
            ).fit_predict_plot(batch_size=32, epochs=epochs)

    


if __name__ == "__main__":
    main()