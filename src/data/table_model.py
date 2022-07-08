from pathlib import PurePosixPath
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, validator
from src.data.storage import get_local_data_path, get_remote_data_path

# TODO: create generic DataObject class, and let Table and TextFile inherit from it


class DataObject(BaseModel):
    path: List[str] = []
    file_name: str = ""
    file_format: str = ".parquet.gzip"
    base_url_position: int = None
    write_func: Optional[Callable]
    kwargs: Dict = {}
    remote: bool = True
    columns: Optional[List[str]]
    local_path: PurePosixPath = None
    remote_path: PurePosixPath = None

    @validator("base_url_position")
    def check_valid_position(cls, v, values):
        if v is not None and len(values["path"]) < v:
            raise Exception(
                "Position for the base url in path is greater than the length of the path itself."
            )
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_path = get_local_data_path(
            self.path, self.file_name, self.file_format
        )
        self.remote_path = get_remote_data_path(
            self.path, self.file_name, self.file_format, self.base_url_position
        )

    class Config:
        arbitrary_types_allowed = True


class Table(DataObject):
    join_on: List[str] = []
    categoricals: List[str] = []

    @validator("join_on", "categoricals")
    def check_in_columns(cls, v, values):
        for col in v:
            if col not in values["columns"]:
                raise Exception(f"{col} is not part of the columns")
        return v
