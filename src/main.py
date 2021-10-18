from pandas import read_csv

from root_locus_nn import RootLocusNN

def main():
    data = read_csv("data/data.csv")
    
    RootLocusNN(
        data=data, 
        input_keys=["uoln", "ucld 0", "ucld 1", "ucld 2"],
        output_keys=["pd -r", "pd -i"],
        standardScaler=True
    ).fit_predict_plot(batch_size=10, epochs=5)
    

if __name__ == "__main__":
    main()