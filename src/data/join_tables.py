from pathlib import PurePosixPath
from typing import List, Set

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


def join(
    tables: List[Table],
    opath: PurePosixPath,
    remote: bool = False,
    # TODO: add argument train_dev_test_ratio
) -> None:
    assert len(tables) >= 2
    categoricals = {category for table in tables for category in table.categoricals}
    ddf = read_categorical_parquet(tables[0], categoricals)
    for i in range(1, len(tables)):
        temp_ddf = read_categorical_parquet(tables[i], categoricals)
        update_categories(ddf, temp_ddf, categoricals)
        cols = [col for col in tables[0].join_on if col in tables[i].join_on]
        ddf = dd.merge(left=ddf, right=temp_ddf, how="left", on=cols)
    # TODO: check correctedness of argument asserting sum == 1
    # TODO: add categorical column "split" and fill it in with 0, 1, 2 (train, dev, test)
    dd_write_parquet(
        opath, ddf, remote, partition_on=["city", "zone"]
    )  # TODO: partition on split as first partition


if __name__ == "__main__":
    join(TABLES, OUTPUT_PATH)
