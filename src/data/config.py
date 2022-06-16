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
    kwargs={},  # TODO
)
