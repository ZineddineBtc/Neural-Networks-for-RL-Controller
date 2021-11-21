from pandas import read_csv

from root_locus_nn import RootLocusNN

def main():
    data = read_csv("data/data.csv")
    model = RootLocusNN(
        data=data, 
        scale=True,
        hiddenLayers_count=2,
        loss="mse",
        test_split=0.2
    )
    model.fit_predict_plot(output_folder="./main_results/", toSave=True,
                            batch_size=32, epochs=35, predictions_title="")
            




if __name__ == "__main__":
    main()