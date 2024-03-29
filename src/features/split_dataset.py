from typing import Tuple

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd


def _check_train_dev_test_ratio(
    train_ratio: float = 0.7,
    dev_ratio: float = None,
    test_ratio: float = None,
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


def _get_splitter(
    n, train_dev_test_ratio: Tuple[float] = (), seed: int = 42
) -> np.ndarray:
    train_ratio, dev_ratio, test_ratio = _check_train_dev_test_ratio(
        *train_dev_test_ratio
    )
    train_n = int(n * train_ratio)
    dev_n = int(n * dev_ratio)
    test_n = n - train_n - dev_n
    rng = np.random.default_rng(seed=seed)
    splitter = rng.permutation(
        np.hstack(
            [
                np.zeros(train_n, dtype=np.int8),
                np.ones(dev_n, dtype=np.int8),
                np.ones(test_n, dtype=np.int8) + 1,
            ]
        )
    )
    return splitter


def dd_split_df(
    ddf: dd.DataFrame,
    train_dev_test_ratio: Tuple[float] = (),
    seed: int = 42,
) -> dd.DataFrame:
    n = len(ddf)
    splitter = _get_splitter(n, train_dev_test_ratio, seed)
    lens = ddf.map_partitions(len).compute().tolist()
    ddf = ddf.assign(split=da.from_array(splitter, chunks=lens))
    ddf.split = ddf.split.astype("category").cat.rename_categories(
        {0: "train", 1: "dev", 2: "test"}
    )
    return ddf


def pd_split_df(
    df: pd.DataFrame,
    train_dev_test_ratio: Tuple[float] = (),
    seed: int = 42,
) -> pd.DataFrame:
    n = len(df)
    splitter = _get_splitter(n, train_dev_test_ratio, seed)
    df = df.assign(
        split=pd.Categorical.from_codes(splitter, categories=["test", "dev", "train"])
    )
    return df
