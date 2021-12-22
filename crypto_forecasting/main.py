from crypto_forecasting.target import ResidualizeMarket
from root_path import ROOT_PATH
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import moment
idx = pd.IndexSlice


def main():
    input_path = ROOT_PATH / "crypto_forecasting" / "inputs"
    # data_set = pd.read_csv(input_path / "train.csv")
    # data_set["timestamp"] = [datetime.fromtimestamp(x) for x in data_set["timestamp"]]
    # data_set.to_pickle(ROOT_PATH / "crypto_forecasting" / "inputs" / "train.pkl")
    # print(data_set.head)
    use_cols = ["Asset_ID", "timestamp", "Volume", "VWAP", "Target"]
    data_set = pd.read_pickle(input_path / "train.pkl")[use_cols].drop_duplicates(
        ["Asset_ID", "timestamp"]
    )
    data_set["VWAP"] = log_return(data_set["VWAP"])
    # target = ResidualizeMarket(asset_1, "Close", 15)

    for asset_id in range(14):
        asset = data_set.loc[data_set["Asset_ID"] == asset_id].set_index("timestamp").sort_index()
        asset = asset.fillna(method="pad").reindex(
            pd.date_range(asset.index[0], asset.index[-1], freq="1min"),
            method="pad",
        )
        modified_data = []

        while True:
            cur_date = asset.index[0]
            mask = asset.index.date == cur_date.date()
            df = asset[mask]
            asset.drop(df.index, inplace=True)
            try:
                # get the 15th minute target on the next day
                target = asset["Target"].iloc[14]
            except Exception:
                break
            moments = moment(df["VWAP"], [1, 2, 3, 4])
            modified_data.append(list(moments) + [target])

        np.savetxt(input_path / "modified_data" / f"asset_{asset_id}.csv", np.array(modified_data), delimiter=",")


def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)


if __name__ == '__main__':
    main()
