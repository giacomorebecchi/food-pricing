import os
import pprint
from functools import partial
from time import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities.parsing import AttributeDict
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Categorical, Integer, Real
from torch.utils.data import DataLoader
from torchvision.models import resnet152
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from xgboost import XGBRegressor

from .nlp.pretrained_bert import PreTrainedBERT
from .utils.data import FoodPricingDataset
from .utils.storage import get_local_models_path


class XGBBaseModel:
    def __init__(self, **kwargs):
        self.save_hyperparameters(kwargs)
        self.txt_transform = self._build_txt_transform()
        self.img_transform = self._build_img_transform()
        self.y_train, self.X_train = self._build_dataset("train")
        self.y_dev, self.X_dev = self._build_dataset("dev")
        self.y_test, self.X_test = self._build_dataset("test")

    def save_hyperparameters(self, kwargs: Dict[str, Any]) -> None:
        self.hparams = AttributeDict(kwargs)
        self._add_model_specific_hparams()
        self._add_default_hparams()

    def fit(self) -> None:
        reg = self._build_model()

        if self.hparams.bayes_search:
            self._prepare_data()
            self.optimizer = self._build_optimizer(estimator=reg)
            self._run_bayes_search()

        else:
            params = self._load_reg_params()
            pass

    def _add_default_hparams(self) -> None:
        default_params = {
            "random_state": 42,
            # load data if it has already been saved
            "load_data": True,
            # dataloader args
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
            # BayesSearchCV
            "bayes_search": True,
            "total_time_stopper": 60 * 20,
            "n_splits": 5,
            # XGBRegressor params
            "booster": "gbtree",
            "objective": "reg:squarederror",
            "tree_method": "hist",
        }
        self.hparams.update({**default_params, **self.hparams})

    def _add_model_specific_hparams(self) -> None:
        pass

    def _build_model(self) -> XGBRegressor:
        reg = XGBRegressor(
            random_state=self.hparams.random_state,
            booster=self.hparams.booster,
            objective=self.hparams.objective,
            tree_method=self.hparams.tree_method,
        )
        return reg

    def _build_optimizer(self, estimator: XGBRegressor) -> BayesSearchCV:
        optimizer = BayesSearchCV(
            estimator=estimator,
            search_spaces=self._define_search_spaces(),
            scoring=self._define_scoring(),
            cv=self.cv_strategy,
            n_iter=120,  # max number of trials
            n_points=1,  # number of hyperparameter sets evaluated at the same time
            n_jobs=1,  # number of jobs
            iid=False,  # if not iid it optimizes on the cv score
            return_train_score=False,
            verbose=2,
            refit=False,
            optimizer_kwargs={
                "base_estimator": "GP"
            },  # optmizer parameters: we use Gaussian Process (GP)
            random_state=self.hparams.random_state,  # random state for replicability
        )
        return optimizer

    def _build_txt_transform(self):
        pass

    def _build_img_transform(self):
        pass

    def _build_dataset(self, split: str) -> Tuple[np.ndarray]:
        data_path = self._get_data_path(split)
        if self.hparams.load_data and os.path.exists(data_path):
            y_X = np.load(data_path)
        else:
            dataset = FoodPricingDataset(
                img_transform=self.img_transform,
                txt_transform=self.txt_transform,
                split=split,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                shuffle=self.hparams.shuffle and split == "train",
                num_workers=self.hparams.loader_workers,
            )
            y_X = self._get_data(dataloader, split)
            np.save(data_path, y_X)
        return y_X[:, 0], y_X[:, 1:]  # y, X

    def _prepare_data(self) -> None:
        self.X = np.vstack([self.X_train, self.X_dev])
        self.y = np.hstack([self.y_train, self.y_dev])

        if self.hparams.n_splits > 1:
            y_stratified = pd.cut(
                pd.Series(self.y).rank(method="first"), bins=10, labels=False
            )

            skf = StratifiedKFold(
                n_splits=self.hparams.n_splits,
                shuffle=True,
                random_state=self.hparams.random_state,
            )

            self.cv_strategy = list(skf.split(self.X, y_stratified))
        else:
            n_train = len(self.y_train)
            n_dev = len(self.y_dev)
            self.cv_strategy = [
                (np.arange(n_train), np.arange(n_train, n_train + n_dev))
            ]

    def _get_data_path(self, split: str) -> str:
        return get_local_models_path(
            path=["data"], model=self, file_name=split, file_format=".npy"
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

    def _define_callbacks(self) -> List:
        overdone_control = DeltaYStopper(delta=1)
        time_limit_control = DeadlineStopper(total_time=self.hparams.total_time_stopper)
        callbacks = [
            overdone_control,
            time_limit_control,
        ]
        return callbacks

    def _define_search_spaces(self):
        search_spaces = {
            "learning_rate": Real(0.01, 1.0, "uniform"),
            "max_depth": Integer(2, 12),
            "subsample": Real(0.1, 1.0, "uniform"),
            "colsample_bytree": Real(
                0.1, 1.0, "uniform"
            ),  # subsample ratio of columns by tree
            "reg_lambda": Real(1e-9, 100.0, "uniform"),  # L2 regularization
            "reg_alpha": Real(1e-9, 100.0, "uniform"),  # L1 regularization
            "n_estimators": Integer(50, 5000),
        }
        return search_spaces

    def _define_scoring(self):
        return make_scorer(
            partial(mean_squared_error, squared=False), greater_is_better=False
        )

    def _run_bayes_search(self):
        start_time = time()
        callbacks = self._define_callbacks()
        self.optimizer.fit(self.X, self.y, callback=callbacks)

        self._report_bayes_search(start_time)

    def _report_bayes_search(self, start_time):
        df = pd.DataFrame(self.optimizer.cv_results_)
        best_score = self.optimizer.best_score_
        best_score_std = df.iloc[self.optimizer.best_index_].std_test_score
        best_params = self.optimizer.best_params_

        print(
            (
                "XGBoost Regression took %.2f seconds,  candidates checked: %d, best CV"
                + "score: %.3f \u00B1"
                + " %.3f"
            )
            % (
                time() - start_time,
                len(self.optimizer.cv_results_["params"]),
                best_score,
                best_score_std,
            )
        )
        print("Best parameters:")
        pprint.pprint(best_params)


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
