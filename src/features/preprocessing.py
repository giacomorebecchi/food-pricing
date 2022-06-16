from pathlib import PurePosixPath
from typing import List, Union

from ..data.storage import dd_read_parquet, dd_write_parquet
from ..data.table_model import Table


def prepare_dataset(
    opath: PurePosixPath,
    remote: bool = False,
    raw_table: Table = Table(),
    columns: Union[List[str], None] = None,
    drop_noimg: bool = True,
    drop_nodescription: bool = False,
) -> None:
    path = raw_table.remote_path if raw_table.remote else raw_table.local_path
    ddf = dd_read_parquet(path, raw_table.remote, columns)
    if drop_noimg:
        ddf = ddf.dropna(subset=["imgPath"])
    if drop_nodescription:
        ddf = ddf.dropna(subset=["description"])
    ddf["txt"] = ddf["name"] + " " + ddf["description"]
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
    ddf = ddf.set_index("id", partition_size="10MB")
    dd_write_parquet(opath, ddf, remote)
