from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats.distributions import uniform
import numpy as np
import pandas as pd
from root_path import ROOT_PATH


class Solution():

    def main(self):
        input_path = ROOT_PATH / "crypto_forecasting/inputs/"
        results_df = pd.DataFrame(0, index=[f"asset {i}" for i in range(14)], columns=["in sample", "out of sample"])
        for asset_id in range(1, 2):
            try:
                asset = pd.read_csv(input_path / f"_modified_data/asset_{asset_id}.csv", header=None).iloc[:-1, :]
                asset.iloc[:, -1] = asset.iloc[:, -1] / asset.iloc[:, -1].mean()  # normalizing the volume
                moments = pd.read_csv(input_path / f"moments/asset_{asset_id}.csv", header=None)
                df = pd.concat([asset, moments], axis=1).dropna()
                x = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                # x_train, x_test, y_train, y_test = self.split_data(x, y, test_size=0.1)
                model = RandomForestRegressor(n_estimators=42, warm_start=True)

                param_dist = {
                    'bootstrap': [True, False],
                    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                    'max_features': ['auto', 'sqrt'],
                    'min_samples_leaf': [1, 2, 4],
                    'min_samples_split': [2, 5, 10],
                    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
                }
                rf_random = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=3,
                                               verbose=2, random_state=420, n_jobs=-1, scoring="r2")

                rf_random.fit(x_train, y_train)
                print(rf_random.best_params_)
                print(rf_random.best_score_)
                # results_df.loc[f"asset {asset_id}", "in sample"] = model.score(x_train, y_train)
                # results_df.loc[f"asset {asset_id}", "out of sample"] = model.score(x_test, y_test)
            except ValueError:
                print(f"asset {asset_id} skipped.")
                continue

        print(results_df)

    @staticmethod
    def split_data(x, y, test_size=0.2):
        n_rows = int(len(x.index) * test_size)
        x_train, x_test = x.iloc[:n_rows, :], x.iloc[n_rows:, :]
        y_train, y_test = y.iloc[:n_rows], y.iloc[n_rows:]
        return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    solution = Solution()
    solution.main()
