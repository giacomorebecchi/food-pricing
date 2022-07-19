import glob
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


def get_best_checkpoint_path(
    model_class: pl.LightningModule = None,
    metric: str = "val_loss",
    asc: bool = True,
) -> str:
    model_name = model_class.__name__ if model_class is not None else ""
    current_path = PurePosixPath(__file__).parent
    parent_path = current_path.parent.parent.parent.joinpath("models", model_name, "*")
    checkpoints = [
        path
        for el in glob.glob(str(parent_path))
        if os.path.isfile(path := PurePosixPath(el)) and path.suffix == ".ckpt"
    ]
    path_score = {
        str(path): float(path.stem.split("=")[-1])
        for path in checkpoints
        if metric in path.stem
    }
    if asc:
        best_checkpoint_path = min(path_score, key=path_score.get)
    else:
        best_checkpoint_path = max(path_score, key=path_score.get)
    return best_checkpoint_path
