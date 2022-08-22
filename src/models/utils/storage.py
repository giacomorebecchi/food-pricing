import glob
import os
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Dict, List, Optional, TypeVar, Union

import pandas as pd
import pytorch_lightning as pl
import yaml

from ...definitions import ROOT_DIR


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
    metric: str = "avg_val_loss",
    asc: bool = True,
    file_format: str = ".ckpt",
) -> str:
    model_name = model_class.__name__ if model_class is not None else ""
    current_path = PurePosixPath(__file__).parent
    parent_path = current_path.parent.parent.parent.joinpath("models", model_name, "*")
    checkpoints = [
        path
        for el in glob.glob(str(parent_path))
        if os.path.isfile(path := PurePosixPath(el)) and path.suffix == file_format
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


def get_run_id() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc, microsecond=0).isoformat()


def store_submission_frame(
    submission_frame: pd.DataFrame,
    model_name: str,
    run_id: Optional[str] = None,
) -> None:
    if run_id is None:
        run_id = get_run_id()
    current_path = PurePosixPath(__file__).parent
    submissions_path = current_path.parent.parent.parent.joinpath(
        "submissions",
        run_id,
    )
    if not os.path.exists(submissions_path):
        print(f"Path {submissions_path} did not exist. Created it.")
        os.makedirs(submissions_path, exist_ok=False)
    path = submissions_path.joinpath(model_name).with_suffix(".csv")
    submission_frame.to_csv(path)


PARAM_TYPES = (
    str,
    int,
    float,
    bool,
    type(None),
)
ParamValue = TypeVar("ParamValue", *PARAM_TYPES)


def get_hparams() -> Dict[str, Union[ParamValue, List[ParamValue]]]:
    hparams_fname = str(PurePosixPath(ROOT_DIR).joinpath("hparams.yml"))
    hparams = yaml.safe_load(open(hparams_fname))
    try:
        assert isinstance(hparams, dict)
        for k, val in hparams.items():
            assert isinstance(k, str)
            if isinstance(val, list):
                for v in val:
                    assert isinstance(v, PARAM_TYPES)
            else:
                assert isinstance(val, PARAM_TYPES)
    except AssertionError:
        print(f"Invalid {hparams} of type {type(hparams)}")
    return hparams
