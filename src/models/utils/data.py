from io import BytesIO
from pathlib import PurePosixPath
from typing import Dict, Generator, List

import torch
import yaml
from PIL import Image, ImageFile
from src.data.config import DATASET, IMAGES
from src.data.storage import CONFIG_PATH, dd_read_parquet, get_S3_fs
from torch import is_tensor
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FoodPricingDataset(Dataset):
    def __init__(
        self,
        img_transform,
        txt_transform,
        split: str = None,
    ) -> None:
        self.config: Dict = yaml.safe_load(open(CONFIG_PATH))
        self.dataset_remote = self.config.get("dataset_remote")
        self.img_thumbnails = self.config.get("img_thumbnails", False)
        self.img_remote = self.config.get("img_remote", not self.img_thumbnails)
        if self.img_thumbnails:
            self.get_img = (
                self.get_remote_thumbnail
                if self.img_remote
                else self.get_local_thumbnail
            )
        else:
            self.get_img = (
                self.get_remote_img if self.img_remote else self.get_local_img
            )
        dataset_path = (
            DATASET.remote_path if self.dataset_remote else DATASET.local_path
        )
        filters = [[("split", "==", split)]] if split else None
        self.data = dd_read_parquet(
            dataset_path,
            self.dataset_remote,
            filters=filters,
        )

        self.index = self.data.index.compute()
        self.img_transform = img_transform
        self.txt_transform = txt_transform

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self,
        idx,
    ) -> Dict:
        if is_tensor(idx):
            idx = idx.tolist()
        idx = self.index[idx]
        ddf = self.data.loc[idx, :].compute()
        img_path = ddf.loc[idx, "imgPath"]
        img = self.get_img(
            local_path=img_path,
            remote_path=img_path,
            name=idx,
        )
        txt = ddf.loc[idx, "txt"]
        lat = ddf.loc[idx, "lat"]
        lon = ddf.loc[idx, "lon"]
        label = ddf.loc[idx, "price_fractional"]
        return {
            "img": self.img_transform(img),
            "txt": self.txt_transform(txt),
            "coords": torch.Tensor([lat, lon]),
            "label": torch.Tensor([label]),
        }

    def get_local_img(
        self, local_path: str = "", remote_path: str = "", name: str = ""
    ) -> Image:
        return Image.open(local_path).convert("RGB")

    def get_remote_img(
        self, local_path: str = "", remote_path: str = "", name: str = ""
    ) -> Image:
        S3 = get_S3_fs()
        with S3.open(remote_path, mode="rb") as f:
            img_bytes = f.read()
        return Image.open(BytesIO(img_bytes)).convert("RGB")

    def get_remote_thumbnail(
        self, local_path: str = "", remote_path: str = "", name: str = ""
    ) -> Image:
        path = PurePosixPath(IMAGES.remote_path).joinpath(name).with_suffix(".jpg")
        S3 = get_S3_fs()
        with S3.open(path, mode="rb") as f:
            img_bytes = f.read()
        return Image.open(BytesIO(img_bytes)).convert("RGB")

    def get_local_thumbnail(
        self, local_path: str = "", remote_path: str = "", name: str = ""
    ) -> Image:
        path = PurePosixPath(IMAGES.local_path).joinpath(name).with_suffix(".jpg")
        return Image.open(path)

    def iter_txt(self, idxs: List[int]) -> Generator:
        for idx in idxs:
            idx = self.index[idx]
            yield self.data.loc[idx, "txt"].compute().values[0]

    def set_txt_transform(self, txt_transform) -> None:
        self.txt_transform = txt_transform

    def set_img_transform(self, img_transform) -> None:
        self.img_transform = img_transform
