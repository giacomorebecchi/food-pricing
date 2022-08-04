import shutil
from pathlib import PurePosixPath

from src.data.config import DATASET, FULL_TABLE
from src.data.config_interim import COORDINATES_TABLE, IMAGES_TABLE, ITEMS_TABLE
from src.data.dataobj_model import Table
from src.data.storage import exists, get_S3_fs


def clean_table(table: Table):
    if exists(table.remote_path, local=False):
        S3 = get_S3_fs()
        S3.rm(str(table.remote_path), recursive=True, maxdepth=None)
    if exists(table.local_path, local=True):
        shutil.rmtree(table.local_path)


if __name__ == "__main__":
    for table in [COORDINATES_TABLE, IMAGES_TABLE, ITEMS_TABLE, FULL_TABLE, DATASET]:
        clean_table(table)
