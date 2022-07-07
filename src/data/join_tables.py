from pathlib import PurePosixPath
from typing import List, Set, Tuple

import dask.dataframe as dd
from src.data.config import COORDINATES_TABLE, IMAGES_TABLE, ITEMS_TABLE
from src.data.storage import dd_read_parquet, dd_write_parquet, get_local_data_path
from src.data.table_model import Table

TABLES = [ITEMS_TABLE, COORDINATES_TABLE, IMAGES_TABLE]
for table in TABLES:
    table.remote = False

OUTPUT_PATH = get_local_data_path(
    path=["processed"],
    file_name="dataset",
    file_format=".parquet.gzip",
)


def update_categories(
    ddf_left: dd.DataFrame,
    ddf_right: dd.DataFrame,
    categorical_cols: Set[str],
) -> None:
    for col in categorical_cols:
        ddf_left[col] = ddf_left[col].cat.add_categories(
            ddf_right[col].cat.categories.difference(ddf_left[col].cat.categories)
        )
        ddf_right[col] = ddf_right[col].cat.add_categories(
            ddf_left[col].cat.categories.difference(ddf_right[col].cat.categories)
        )


def read_categorical_parquet(
    table: Table,
    categoricals: Set[str],
) -> dd.DataFrame:
    path = table.remote_path if table.remote else table.local_path
    ddf = dd_read_parquet(path, table.remote, table.columns)
    return ddf.categorize(columns=list(categoricals))


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


def join(
    tables: List[Table],
    opath: PurePosixPath,
    remote: bool = False,
    train_dev_test_ratio: Tuple[float] = (),
) -> None:
    assert len(tables) >= 2
    categoricals = {category for table in tables for category in table.categoricals}
    ddf = read_categorical_parquet(tables[0], categoricals)
    for i in range(1, len(tables)):
        temp_ddf = read_categorical_parquet(tables[i], categoricals)
        update_categories(ddf, temp_ddf, categoricals)
        cols = [col for col in tables[0].join_on if col in tables[i].join_on]
        ddf = dd.merge(left=ddf, right=temp_ddf, how="left", on=cols)
    train_ratio, dev_ratio, test_ratio = _check_train_dev_test_ratio(
        *train_dev_test_ratio
    )
    # TODO: add categorical column "split" and fill it in with 0, 1, 2 (train, dev, test)
    dd_write_parquet(
        opath, ddf, remote, partition_on=["city", "zone"]
    )  # TODO: partition on split as first partition


if __name__ == "__main__":
    join(TABLES, OUTPUT_PATH)
