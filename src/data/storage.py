import json
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Dict, List

import pandas as pd
import pyarrow.fs
import s3fs
from dotenv import load_dotenv
from fsspec import AbstractFileSystem
from src.data.settings import get_S3_settings

# load environment variables
load_dotenv()


@lru_cache()
def get_S3_pyarrow_fs() -> pyarrow.fs.S3FileSystem:
    S3 = pyarrow.fs.S3FileSystem(**get_S3_settings().format_settings_pyarrow_fs())
    return S3


@lru_cache()
def get_S3_fs() -> s3fs.S3FileSystem:
    S3 = s3fs.S3FileSystem(**get_S3_settings().format_settings_s3fs())
    return S3


def write_dict(fs: AbstractFileSystem, path: str, obj: Dict) -> None:
    if not isinstance(obj, dict):
        raise Exception("Data type must be dict")
    file_format = PurePosixPath(path).suffix
    with fs.open(path, mode="wb") as file:
        if file_format == ".json":
            obj = json.dumps(obj).encode()
            file.write(obj)
        else:
            raise Exception(f"File format {file_format} not supported for type dict")


def write_list(fs: AbstractFileSystem, path: str, obj: List) -> None:
    if not isinstance(obj, list):
        raise Exception("Data type must be list")
    file_format = PurePosixPath(path).suffix
    with fs.open(path, mode="wb") as file:
        if file_format == ".jsonlines":
            pd.DataFrame(obj).to_json(file, lines=True, orient="records")
        elif file_format == ".csv":
            df = pd.json_normalize(obj, sep="_")
            df.to_csv(file)
        else:
            raise Exception(f"File format {file_format} not supported for type list")


def write_bytes(fs: AbstractFileSystem, path: str, obj: bytes) -> None:
    if not isinstance(obj, bytes):
        if isinstance(obj, str):
            obj = obj.encode()
        else:
            raise Exception("Data type must be bytes")
    file_format = PurePosixPath(path).suffix
    with fs.open(path, mode="wb") as file:
        if file_format in [".xml", ".html", ".txt", ".jpg", ".jpeg", ".webp", ".json"]:
            file.write(obj)
        else:
            raise Exception(f"File format {file_format} not supported for type bytes")


def load_bytes(fs: AbstractFileSystem, path: str) -> bytes:
    with fs.open(path, mode="rb") as file:
        obj = file.read()
    return obj


def build_path(*args: str) -> str:
    path = PurePosixPath(get_S3_settings().BUCKET).joinpath(*args)
    return str(path)


def build_output_path(path: str, sub_map: Dict) -> str:
    for key, value in sub_map.items():
        path = path.replace(key, value, 1)
    return path
