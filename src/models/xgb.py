import itertools
import json
import os
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.utils.data import DataLoader
from torchvision.models import resnet152
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from xgboost import DMatrix
from xgboost import train as xgb_train

from .dual_encoding.pretrained_clip import PreTrainedCLIP
from .nlp.pretrained_bert import PreTrainedBERT
from .utils.data import FoodPricingDataset
from .utils.storage import get_local_models_path


class XGBBaseModel:
    def __init__(self, **kwargs):
        self.save_hyperparameters(kwargs)

        # build dual model, which has the precedence over other transformers
        if self.hparams.dual_model:
            self.dual_transform = self._build_dual_transform()

        # build transform models
        self.txt_transform = self._build_txt_transform()
        self.img_transform = self._build_img_transform()

        # build the datasets
        self.d_train = self._build_dataset("train")
        self.d_dev = self._build_dataset("dev")
        self.d_test = self._build_dataset("test")

    def save_hyperparameters(self, kwargs: Dict[str, Any]) -> None:
        self.hparams = AttributeDict(kwargs)
        self._add_model_specific_hparams()
        self._add_default_hparams()

    def fit(self):
        params_grid = self._calculate_grid()
        self.grid_search_results = {}
        self.best_score = None
        for i, params in enumerate(params_grid):
            self.grid_search_results[i] = {"params": params}

            regressor = xgb_train(
                params=params,
                dtrain=self.d_train,
                num_boost_round=self.hparams.num_round,
                evals=[(self.d_dev, "dev")],
                early_stopping_rounds=self.hparams.early_stopping_rounds,
                verbose_eval=False,
            )
            self.grid_search_results[i]["best_score"] = regressor.best_score
            if self.best_score is None or self.best_score > regressor.best_score:
                best_iter = i
                self.best_score = regressor.best_score
                self.best_model = regressor[: regressor.best_iteration + 1]
                print(
                    "Found a new optimal parametrization with score: "
                    f"{self.best_score}"
                )

        self.best_params = self.grid_search_results[best_iter]["params"]
        print("The optimal model has the following parameters: ", self.best_params)
        self._store_model()

    def _add_default_hparams(self) -> None:
        default_params = {
            # No dual encoding by default
            "dual_model": False,
            # load data if it has already been saved
            "load_data": True,
            # DataLoader args
            "batch_size": 32,
            "loader_workers": 8,
            "shuffle": False,
            # img_transform hyperparameters
            "img_dim": 224,
            # all torchvision models expect the same
            # normalization mean and std
            # https://pytorch.org/vision/stable/models.html
            "img_mean": [0.485, 0.456, 0.406],
            "img_std": [0.229, 0.224, 0.225],
            # Best hyperparameters settings
            "max_iter": 50,
            "max_seconds": 60 * 60,
            # "growpolicy": ["depthwise", "lossguide"]
            # XGB train hyperparameters
            "num_round": 100,
            "early_stopping_rounds": 5,
        }
        xgb_default_params = {
            # XGBRegressor hyperparameters
            "booster": "gbtree",
            "tree_method": "hist",
            "colsample_bytree": [0.5, 0.8, 1],
            "objective": "reg:squarederror",
            "eta": [0.3, 0.5],
            "max_depth": [6, 10, 15],
            "subsample": [0.8, 1],
        }
        self.xgb_keys = xgb_default_params.keys()
        self.hparams.update({**default_params, **xgb_default_params, **self.hparams})

    def _add_model_specific_hparams(self) -> None:
        pass

    def _build_txt_transform(self) -> Callable:
        return lambda _: _

    def _build_img_transform(self) -> Callable:
        return lambda _: _

    def _build_dual_transform(self):
        return None

    def _build_dataset(self, split: str) -> DMatrix:
        data_path = self._get_data_path(split)
        if self.hparams.load_data and os.path.exists(data_path):
            ar = DMatrix(data_path)
        else:
            dataset = FoodPricingDataset(
                img_transform=self.img_transform,
                txt_transform=self.txt_transform,
                dual_transform=self.dual_transform if self.hparams.dual_model else None,
                split=split,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                shuffle=self.hparams.shuffle and split == "train",
                num_workers=self.hparams.loader_workers,
            )
            y_X = self._get_data(dataloader, split)
            ar = DMatrix(y_X[:, 1:], label=y_X[:, 0])
            ar.save_binary(data_path)
        return ar

    def _calculate_grid(self) -> List[Dict]:
        params = {key: self.hparams[key] for key in self.xgb_keys}
        keys, values = zip(*params.items())
        values = [[v] if not isinstance(v, (list, tuple)) else v for v in values]
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        n = len(permutations_dicts)
        print(f"Found {n} different params combinations")
        if n > (n_max := self.hparams.max_iter):
            print(
                "The number of different params combinations is greater than "
                f"the limit you specified through the parameter 'max_iter' ({n_max})."
            )
            permutations_dicts = permutations_dicts[:n_max]
            print(f"Exploring only the first {n_max} combinations.")
        return permutations_dicts

    def _get_data_path(self, split: str) -> str:
        return get_local_models_path(
            path=["data"], model=self, file_name=split, file_format=".buffer"
        )

    def _get_data(self, dataloader: DataLoader, split: str) -> np.ndarray:
        ar = torch.vstack(
            [
                torch.hstack(
                    (
                        batch["label"],
                        torch.squeeze(batch["img"]),
                        torch.squeeze(batch["txt"]),
                    )
                )
                for batch in tqdm(dataloader, desc=f"Creating {split} dataset: ")
            ]
        ).numpy()
        return ar

    def _store_model(self) -> None:
        fname = self.hparams.objective + str(round(self.best_score, 3))
        model_path = get_local_models_path(
            path=[], model=self, file_name="model_" + fname, file_format=".json"
        )
        self.best_model.save_model(model_path)

        params = {**self.hparams, **self.best_params}
        params_path = get_local_models_path(
            path=[], model=self, file_name="params_" + fname, file_format=".json"
        )
        with open(params_path, mode="w") as f:
            json.dump(params, f)


class XGBBERTResNet152(XGBBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_txt_transform(self):
        model_kwargs = {"pretrained_model_name_or_path": self.hparams.bert_model}
        tokenizer_kwargs = {
            "pretrained_model_name_or_path": self.hparams.tokenizer_bert_model
        }
        language_transform = PreTrainedBERT(
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.hparams.update(
            {
                "embedding_dim": language_transform.encoder_features,
            }
        )
        return language_transform

    def _build_img_transform(self):
        module = resnet152(weights="DEFAULT")
        img_dim = self.hparams.img_dim
        transformer = Compose(
            [
                Resize(size=(img_dim, img_dim)),
                ToTensor(),
                Normalize(mean=self.hparams.img_mean, std=self.hparams.img_std),
            ]
        )
        for param in module.parameters():
            param.requires_grad = False
        return lambda img: module(torch.unsqueeze(transformer(img), 0))

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "bert_model": "dbmdz/bert-base-italian-xxl-uncased",
            "tokenizer_bert_model": self.hparams.get(
                "bert_model", "dbmdz/bert-base-italian-xxl-uncased"
            ),
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})


class XGBCLIP(XGBBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_img_transform(self):
        img_dim = self.hparams.img_dim
        transformer = Compose(
            [
                Resize(size=(img_dim, img_dim)),
                ToTensor(),
                Normalize(mean=self.hparams.img_mean, std=self.hparams.img_std),
            ]
        )
        return transformer

    def _build_dual_transform(self):
        model_kwargs = {"pretrained_model_name_or_path": self.hparams.clip_model}
        processor_kwargs = {
            "pretrained_model_name_or_path": self.hparams.processor_clip_model
        }
        clip = PreTrainedCLIP(
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            img_feature_dim=self.hparams.language_feature_dim,
            txt_feature_dim=self.hparams.vision_feature_dim,
            return_tensors=None,
        )
        self._update_clip_hparams(clip)
        return clip

    def _update_clip_hparams(self, clip: PreTrainedCLIP) -> None:
        processor_config = clip.processor.feature_extractor
        self.hparams.update(
            {
                "img_dim": processor_config.crop_size,
                "img_mean": processor_config.image_mean,
                "img_std": processor_config.image_std,
            }
        )
        self.hparams.update(
            {
                "projection_dim": clip.encoder_features,
            }
        )

    def _add_model_specific_hparams(self) -> None:
        model_specific_hparams = {
            "dual_model": True,
            "clip_model": "clip-italian/clip-italian",
            "processor_clip_model": self.hparams.get(
                "clip_model", "clip-italian/clip-italian"
            ),  # Default is same as "clip_model" param
        }
        self.hparams.update({**model_specific_hparams, **self.hparams})
