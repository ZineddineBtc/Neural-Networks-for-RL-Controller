from pandas import read_csv

from models_dict import possible_inputs, possible_outputs
from root_locus_nn import RootLocusNN

def main():
    data = read_csv("data/data.csv")
    
    epochs = 5

    for input in possible_inputs:
        for output in possible_outputs:
            for b in [True, False]:
                RootLocusNN(
                    model_name=input["name"]+"--"+output["name"],
                    data=data, 
                    input_keys=input["keys"],
                    output_keys=output["keys"],
                    standardScaler=b
                ).fit_predict_plot(batch_size=10, epochs=epochs)

    


if __name__ == "__main__":
    main()