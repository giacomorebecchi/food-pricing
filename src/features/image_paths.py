import re
from typing import List

import dask.bag as db
from dask import delayed
from dotenv import load_dotenv
from fsspec import AbstractFileSystem
from src.data.settings import get_S3_settings
from src.data.storage import build_path, get_local_data_path, get_S3_fs

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
    rf"(?P<imgPath>{get_S3_settings().BUCKET}/data/images/(?:[a-zA-Z.]+)/store/(?P<city>[a-zA-Z\-]+)/(?P<zone>[a-zA-Z\-]+)/(?:[a-zA-z0-9\-]+)/(?:[T0-9\.\:\-]+)/(?P<store>[a-zA-Z0-9\-]+)-(?P<menuRow>[0-9]+)+.(?P<format>[a-zA-Z0-9]+))$"
)
OUTPUT_PATH = get_local_data_path(path=["interim", "images"])


def get_children(
    fs: AbstractFileSystem, parent_dir: str = "", child_path: str = "/**/*.*"
):
    path = parent_dir + child_path
    return fs.glob(path)


@delayed
def compute_img(paths: List[str], pattern: re.Pattern):
    return [re.match(pattern, path).groupdict() for path in paths]


def main(
    cities_path: str,
    zones_subpath: str,
    imgs_subpath: str,
    img_path_pattern: re.Pattern,
    opath: str,
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
    ddf.to_parquet(opath, partition_on=["city", "zone"], compression="gzip")


if __name__ == "__main__":
    main(CITIES_PATH, ZONES_SUBPATH, IMGS_SUBPATH, IMG_PATH_PATTERN, OUTPUT_PATH)
