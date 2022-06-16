import functools
import inspect
import os
import pickle
from pathlib import PurePosixPath
from typing import Callable, Dict

from src.data.storage import get_local_data_path
from src.definitions import ROOT_DIR


class FileEmptyError(Exception):
    pass


class Cache:
    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)
        self.file = inspect.getfile(func)
        self.func = func
        self.path = self.get_path()
        self.cache = self.load_cache()

    def __call__(self, *args, **kwargs):
        cache_key = args + tuple(kwargs.items())
        if cache_key in self.cache:
            return self.cache[cache_key]
        else:
            self.cache[cache_key] = result = self.func(*args, **kwargs)
            return result

    def get_path(self) -> PurePosixPath:
        relative_path = os.path.relpath(self.file, ROOT_DIR)
        cache_path = get_local_data_path(
            ["interim", "cache", relative_path],
            file_name=self.__name__,
            file_format=".pickle",
        )
        return cache_path

    def load_cache(self) -> Dict:
        try:
            with open(self.path, mode="rb") as file:
                pickled_cache = file.read()
                if not pickled_cache:
                    raise FileEmptyError
                else:
                    cache = pickle.loads(pickled_cache)
        except (FileNotFoundError, FileEmptyError):
            cache = {}
        return cache

    def store_cache(self) -> None:
        with open(self.path, mode="wb") as file:
            pickle.dump(self.cache, file)
