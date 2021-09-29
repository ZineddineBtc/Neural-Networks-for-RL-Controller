from pandas import read_csv

from model_0 import Model_0

def main():
    data = read_csv("data/data.csv")
    
    Model_0(
        model_name="model_0", 
        data=data, 
        input_keys=["uclp 0", "uclp 1", "uclp 2"], 
        output_keys=["pd"]
    ).train_predict_plot(epochs=10)
    

if __name__ == "__main__":
    main()