from kaggle.target import ResidualizeMarket
from root_path import ROOT_PATH
import pandas as pd
idx = pd.IndexSlice


def main():
    input_path = ROOT_PATH / "inputs" / "train.csv"
    df = pd.read_csv(input_path, index_col=["timestamp", "Asset_ID"], nrows=50000)
    asset_1 = df.loc[idx[:, 1], ["Close", "Target"]]
    print(asset_1.head)
    target = ResidualizeMarket(asset_1, "Close", 15)
    print(target[0].head)


if __name__ == '__main__':
    main()
