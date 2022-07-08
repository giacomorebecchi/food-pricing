from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, List, Tuple, Union

import dask.dataframe as dd
from PIL import Image
from s3fs import S3FileSystem

from ..data.storage import dd_read_parquet, get_S3_fs
from ..data.table_model import Table


def save_img(
    s: dd.Series,
    S3: S3FileSystem,
    opath: PurePosixPath,
    size: Tuple[int, int],
) -> None:
    img_opath = PurePosixPath(opath).joinpath(s.name).with_suffix(".jpg")
    with S3.open(s["imgPath"], mode="rb") as f_img:
        img_byte_str = f_img.read()
    with Image.open(BytesIO(img_byte_str)) as im:
        im.thumbnail(size)
        im.save(str(img_opath), "JPEG")


def download_thumbnails(
    opath: PurePosixPath,
    remote: bool = False,
    raw_table: Table = Table(),
    imgPath_column: Union[List[str], None] = "imgPath",
    size: Tuple[int, int] = (450, 450),
) -> None:
    path = raw_table.remote_path if raw_table.remote else raw_table.local_path
    ddf = dd_read_parquet(
        path,
        raw_table.remote,
        columns=[imgPath_column],
    )
    S3 = get_S3_fs()
    _ = ddf.apply(
        lambda row: save_img(row, S3, opath, size), axis=1, meta=(None, "object")
    ).compute()
