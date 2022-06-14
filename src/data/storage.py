import json
import os
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.fs
import s3fs
from dotenv import load_dotenv
from fsspec import AbstractFileSystem

from .settings import get_S3_settings

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


def load_list(fs: AbstractFileSystem, path: str) -> List:
    if not path.endswith(".jsonlines"):
        raise Exception("Function supported with .jsonlines format only.")
    output = []
    with fs.open(path, mode="rb") as file:
        for line in file:
            output.append(json.loads(line))
    return output


def build_path(*args: str) -> PurePosixPath:
    path = PurePosixPath(get_S3_settings().BUCKET).joinpath(*args)
    return path


def build_output_path(path: str, sub_map: Dict) -> PurePosixPath:
    for key, value in sub_map.items():
        path = path.replace(key, value, 1)
    return PurePosixPath(path)


def get_local_data_path(
    path: List, file_name: str = "", file_format: str = ""
) -> PurePosixPath:
    current_path = PurePosixPath(__file__).parent
    data_path = current_path.parent.parent.joinpath("data", *path)
    if not os.path.exists(data_path):
        print(f"Path {data_path} did not exist. Created it.")
        os.makedirs(data_path, exist_ok=False)
    path = data_path.joinpath(file_name).with_suffix(file_format)
    return path


def get_remote_data_path(
    path: List,
    file_name: str = "",
    file_format: str = "",
    base_url_position: int = None,
):
    if base_url_position is not None:
        path.insert(base_url_position, os.environ["BASE_URL"])
    fpath = (
        PurePosixPath("data")
        .joinpath(*path)
        .joinpath(file_name)
        .with_suffix(file_format)
    )
    return fpath


def get_children(
    fs: AbstractFileSystem, parent_dir: str = "", child_path: str = "/**/*.*"
):
    path = parent_dir + child_path
    return fs.glob(path)


def exists(
    fpath: PurePosixPath,
    local: bool = True,
) -> bool:
    try:
        if local:
            return get_local_size(fpath) > 0
        else:
            return get_remote_size(fpath) > 0
    except Exception:  # TODO: capture correct Exception
        return False


def get_local_size(path: PurePosixPath) -> int:
    def get_local_dir_size(path: PurePosixPath) -> int:
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += os.path.get_size(entry)
                elif entry.is_dir():
                    total += get_local_dir_size(entry.path)
        return total

    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        return get_local_dir_size(path)


def get_remote_size(path: PurePosixPath) -> int:
    S3 = get_S3_fs()
    return S3.du(path)
