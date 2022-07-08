from ..features.preprocessing import prepare_dataset
from .join_tables import join
from .table_model import Table

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
    },
)

# TODO: add a TXT_TRAIN object of class TextFile
