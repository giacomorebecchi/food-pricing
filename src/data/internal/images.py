import re
from pathlib import PurePosixPath
from typing import List

import dask.bag as db
import pandas as pd
from dask import delayed
from dotenv import load_dotenv
from PIL import Image
from src.data.settings import get_S3_settings
from src.data.storage import build_path, get_children, get_local_data_path, get_S3_fs

load_dotenv()

CITIES_PATH = build_path(
    "data",
    "images",
    "*",
    "store",
    "*",  # city
)
ZONES_SUBPATH = "/*"

IMGS_SUBPATH = "/**/*.*"

IMG_PATH_PATTERN = re.compile(
    rf"(?P<imgPath>{get_S3_settings().BUCKET}/data/images/(?:[a-zA-Z.]+)/store/(?P<city>[a-zA-Z\-]+)/(?P<zone>[a-zA-Z\-]+)/(?:[a-zA-z0-9\-]+)/(?:[T0-9\.\:\-]+)/(?P<store>[a-zA-Z0-9\-]+)-(?P<menuRow>[0-9]+)+(?P<format>.[a-zA-Z0-9]+))$"
)
OUTPUT_PATH = get_local_data_path(
    path=["interim"], file_name="images", file_format=".parquet.gzip"
)


@delayed
def compute_img(paths: List[str], pattern: re.Pattern):
    return [
        {
            key: int(value) if value.isnumeric() else value
            for key, value in re.match(pattern, path).groupdict().items()
        }
        for path in paths
    ]


def compute_shape(path: str) -> pd.Series:
    S3 = get_S3_fs()
    with S3.open(path, mode="rb") as f:
        a = Image.open(f)
    return pd.Series({"height": a.height, "width": a.width})


def make_images_table(
    cities_path: str,
    zones_subpath: str,
    imgs_subpath: str,
    img_path_pattern: re.Pattern,
    opath: PurePosixPath,
    remote: bool = False,
    get_shapes: bool = False,
) -> None:
    S3 = get_S3_fs()
    zone_paths = [
        zone
        for city in get_children(S3, parent_dir="", child_path=cities_path)
        for zone in get_children(S3, city, zones_subpath)
    ]
    img_paths = [
        delayed(get_children)(S3, zone_path, imgs_subpath) for zone_path in zone_paths
    ]
    img_data = [
        compute_img(img_paths_zone, img_path_pattern) for img_paths_zone in img_paths
    ]
    ddf = db.from_delayed(img_data).to_dataframe()
    if get_shapes:
        ddf_size = ddf.imgPath[ddf.format == ".jpeg"].apply(  # why only .jpeg?
            compute_shape, meta={"height": int, "width": int}
        )
        ddf = ddf.assign(height=ddf_size.height, width=ddf_size.width)
    # TODO: custom to_parquet here
    ddf.to_parquet(opath, partition_on=["city", "zone"], compression="gzip")


if __name__ == "__main__":
    make_images_table(
        CITIES_PATH, ZONES_SUBPATH, IMGS_SUBPATH, IMG_PATH_PATTERN, OUTPUT_PATH
    )
