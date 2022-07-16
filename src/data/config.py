from ..features.images import download_thumbnails
from ..features.preprocessing import prepare_dataset
from ..features.text_file import create_txt
from .dataobj_model import DataObject, Table
from .join_tables import join

FULL_TABLE = Table(
    path=["interim"],
    file_name="dataset",
    file_format=".parquet.gzip",
    base_url_position=1,
    write_func=join,
    kwargs={},
)

DATASET = Table(
    path=["processed"],
    file_name="dataset",
    file_format=".parquet.gzip",
    base_url_position=1,
    write_func=prepare_dataset,
    kwargs={
        "columns": [
            "name",
            "description",
            "imgPath",
            "price_fractional",
            "lat",
            "lon",
            "city",
            "zone",
            "store",
            "menuRow",
        ],
        "drop_noimg": True,
        "drop_nodescription": False,
        "replace_specialchars": True,
        "fillna_description": "EMPTY_DESCRIPTION",
    },
)

TXT_TRAIN = DataObject(
    path=["processed", "txt"],
    file_name="train",
    file_format=".txt",
    base_url_position=1,
    write_func=create_txt,
    kwargs={
        "columns": ["txt"],
        "filters": [
            [
                ("split", "==", "train"),
            ],
        ],
    },
)

IMAGES = DataObject(
    path=["processed", "img"],
    file_name="",
    file_format="",
    write_func=download_thumbnails,
    remote=False,
    kwargs={
        "imgPath_column": "imgPath",
        "size": (450, 450),
    },
)  # TODO: complete IMAGES DataObject
