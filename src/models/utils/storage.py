import os
from pathlib import PurePosixPath
from typing import List

import pytorch_lightning as pl


def get_local_models_path(
    path: List[str],
    model: pl.LightningModule = None,
    file_name: str = "",
    file_format: str = "",
) -> PurePosixPath:
    model_name = type(model).__name__ if model is not None else ""
    current_path = PurePosixPath(__file__).parent
    models_path = current_path.parent.parent.parent.joinpath(
        "models", model_name, *path
    )
    if not os.path.exists(models_path):
        print(f"Path {models_path} did not exist. Created it.")
        os.makedirs(models_path, exist_ok=False)
    path = models_path.joinpath(file_name).with_suffix(file_format)
    return path
