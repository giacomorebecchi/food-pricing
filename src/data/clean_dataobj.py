import os
import shutil

from src.data.config import DATASET, FULL_TABLE, IMAGES, TXT_TRAIN
from src.data.config_interim import COORDINATES_TABLE, IMAGES_TABLE, ITEMS_TABLE
from src.data.dataobj_model import DataObject
from src.data.storage import exists, get_S3_fs


def clean_dataobj(dataobj: DataObject):
    if exists(dataobj.remote_path, local=False):
        S3 = get_S3_fs()
        S3.rm(str(dataobj.remote_path), recursive=True, maxdepth=None)
    if exists(dataobj.local_path, local=True):
        if os.path.isdir(dataobj.local_path):
            shutil.rmtree(dataobj.local_path)
        elif os.path.isfile(dataobj.local_path):
            os.remove(dataobj.local_path)


if __name__ == "__main__":
    for obj in [
        # COORDINATES_TABLE,
        # IMAGES_TABLE,
        # ITEMS_TABLE,
        # FULL_TABLE,
        DATASET,
        TXT_TRAIN,
        # IMAGES,
    ]:
        clean_dataobj(obj)
