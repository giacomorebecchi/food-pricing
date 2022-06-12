from pathlib import PurePosixPath
from typing import Dict, List

import dask.dataframe as dd
from src.data.storage import get_local_data_path

DATA_INFO = [
    {
        "name": "menus",
        "path": get_local_data_path(path=["interim", "menu"]),
        "columns": [
            "name",
            "description",
            "price_fractional",
            "menuRow",
            "city",
            "zone",
            "store",
        ],
    },  # here order matters, menus must be the first
    {
        "name": "coordinates",
        "path": get_local_data_path(
            path=["external", "geopy"],
            file_name="coordinates",
            file_format=".parquet",
        ),
        "columns": ["city", "zone", "lat", "lon"],
    },  # here order matters, coordinates must be the second
    {
        "name": "images",
        "path": get_local_data_path(path=["interim", "images"]),
        "columns": ["imgPath", "store", "menuRow", "city", "zone"],
    },  # here order matters, images must be last
]

JOIN_ON = ["city", "zone", "store", "menuRow"]
CATEGORICALS = ["city", "zone"]
OUTPUT_PATH = get_local_data_path(
    path=["processed"],
    file_name="dataset",
    file_format=".parquet",
)


def update_categories(
    ddf_left: dd.DataFrame,
    ddf_right: dd.DataFrame,
    categorical_cols: List[str],
) -> None:
    for col in categorical_cols:
        ddf_left[col] = ddf_left[col].cat.add_categories(
            ddf_right[col].cat.categories.difference(ddf_left[col].cat.categories)
        )
        ddf_right[col] = ddf_right[col].cat.add_categories(
            ddf_left[col].cat.categories.difference(ddf_right[col].cat.categories)
        )


def read_parquet(
    data_info: Dict,
    categoricals: List[str],
) -> dd.DataFrame:
    ddf = dd.read_parquet(data_info["path"], columns=data_info["columns"])
    return ddf.categorize(columns=categoricals)


def merge(
    data_info: List[Dict],
    on: List[str],
    categoricals: List[str],
    opath: PurePosixPath,
) -> None:
    assert len(data_info) >= 2
    ddf = read_parquet(data_info[0], categoricals)
    for i in range(1, len(data_info)):
        temp_ddf = read_parquet(data_info[i], categoricals)
        update_categories(ddf, temp_ddf, categoricals)
        cols = [col for col in on if (col in ddf.columns) and (col in temp_ddf.columns)]
        ddf = dd.merge(
            left=ddf, right=temp_ddf, how="left", on=cols
        )  # Python>=3.7: dict are ordered
    ddf.to_parquet(opath, partition_on=["city", "zone"], compression="gzip")


if __name__ == "__main__":
    merge(DATA_INFO, JOIN_ON, CATEGORICALS, OUTPUT_PATH)
