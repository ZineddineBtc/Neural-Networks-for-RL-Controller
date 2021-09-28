from pandas import read_csv

from model_0 import Model_0

def main():
    data = read_csv("data/data.csv")

    model_0 = Model_0(model_name="model_0", data=data, unused_columns=[3, 4, 5, 11], output_heads=["pd"])
    model_0.train_predict_plot(epochs=10)

if __name__ == "__main__":
    main()