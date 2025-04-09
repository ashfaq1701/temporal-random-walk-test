import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    input_file = "data/sx-stackoverflow.txt"
    output_file = "data/sx-stackoverflow.csv"

    df = pd.read_csv(input_file, sep=" ", header=None, names=["u", "i", "ts"])

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df['u'] = user_encoder.fit_transform(df['u'])
    df['i'] = item_encoder.fit_transform(df['i'])

    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
