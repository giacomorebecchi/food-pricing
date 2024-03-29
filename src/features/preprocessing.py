from pathlib import PurePosixPath
from typing import List, Tuple, Union

from src.features.split_dataset import dd_split_df

from ..data.dataobj_model import Table
from ..data.storage import dd_read_parquet, dd_write_parquet


def prepare_dataset(
    opath: PurePosixPath,
    remote: bool = False,
    raw_table: Table = Table(),
    columns: Union[List[str], None] = None,
    drop_noimg: bool = True,
    drop_nodescription: bool = False,
    replace_specialchars: bool = True,
    fillna_description: str = "",
    train_dev_test_ratio: Tuple[float] = (),
    seed: int = 42,
) -> None:
    path = raw_table.remote_path if raw_table.remote else raw_table.local_path
    ddf = dd_read_parquet(path, raw_table.remote, columns)
    if drop_noimg:
        ddf = ddf.dropna(subset=["imgPath"])
    if drop_nodescription:
        ddf = ddf.dropna(subset=["description"])
    else:
        ddf["description"] = ddf["description"].fillna(fillna_description)
    ddf["txt"] = ddf["name"] + " " + ddf["description"]
    if replace_specialchars:
        ddf["txt"] = ddf["txt"].str.replace("\n", " ", regex=False)
    ddf["item_id"] = (
        ddf["city"].astype(str)
        + "_"
        + ddf["zone"].astype(str)
        + "_"
        + ddf["store"]
        + "_"
        + ddf["menuRow"].astype(str)
    )
    ddf = ddf.drop(columns=["name", "description", "city", "zone", "store", "menuRow"])
    ddf = ddf.set_index("item_id", partition_size="10MB")
    ddf = dd_split_df(ddf, train_dev_test_ratio, seed)
    dd_write_parquet(opath, ddf, remote, ["split"])
