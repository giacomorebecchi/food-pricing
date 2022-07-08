import shutil

from src.data.config import DATASET, FULL_TABLE
from src.data.config_interim import COORDINATES_TABLE, IMAGES_TABLE, ITEMS_TABLE
from src.data.storage import exists, get_S3_fs
from src.data.table_model import DataObject


def clean_dataobj(dataobj: DataObject):
    if exists(dataobj.remote_path, local=False):
        S3 = get_S3_fs()
        S3.rm(str(dataobj.remote_path), recursive=True, maxdepth=None)
    if exists(dataobj.local_path, local=True):
        shutil.rmtree(dataobj.local_path)


if __name__ == "__main__":
    for table in [COORDINATES_TABLE, IMAGES_TABLE, ITEMS_TABLE, FULL_TABLE, DATASET]:
        clean_dataobj(table)
    # TODO: add txt file here
