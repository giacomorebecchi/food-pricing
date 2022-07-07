from typing import Tuple

import numpy as np
import pandas as pd


def _check_train_dev_test_ratio(
    train_ratio: float = 0.7, dev_ratio: float = None, test_ratio: float = None
) -> Tuple[float]:
    def check_value(var_name: str, val: float) -> None:
        if not (isinstance(val, float) and 0 < val < 1):
            raise ValueError(
                f"Argument {var_name} is invalid. It must be a float between 0 and 1."
            )

    check_value("train_ratio", train_ratio)
    if test_ratio is None:
        if dev_ratio is None:
            # default: dev and test datasets have the same size
            dev_ratio = test_ratio = (1 - train_ratio) / 2
        else:
            check_value("dev_ratio", dev_ratio)
            test_ratio = 1 - train_ratio - dev_ratio
    check_value("test_ratio", test_ratio)
    if abs(train_ratio + dev_ratio + test_ratio - 1) > 1e-07:
        raise ValueError("The sum of the train, dev and test ratio must be equal to 1.")
    return train_ratio, dev_ratio, test_ratio


def split_df(
    df: pd.DataFrame, train_dev_test_ratio: Tuple[float] = (), seed: int = 42
) -> pd.DataFrame:
    train_ratio, dev_ratio, test_ratio = _check_train_dev_test_ratio(
        *train_dev_test_ratio
    )
    n = len(df)
    train_n = int(n * train_ratio)
    dev_n = int(n * dev_ratio)
    rng = np.random.default_rng(seed=seed)
    train_dev = rng.choice(n, train_n + dev_n, replace=False)
    train = train_dev[:train_n]
    dev = train_dev[train_n:]
    df["split"] = "test"
    df.iloc[train, df.columns.get_loc("split")] = "train"
    df.iloc[dev, df.columns.get_loc("split")] = "dev"
    return df
