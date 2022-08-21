from io import BytesIO
from pathlib import PurePosixPath
from typing import Callable, Dict, Generator, Optional

import yaml
from PIL import Image, ImageFile
from src.data.config import DATASET, IMAGES
from src.data.storage import CONFIG_PATH, dd_read_parquet, get_S3_fs, pd_read_parquet
from torch import Tensor, is_tensor
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FoodPricingDataset(Dataset):
    def __init__(
        self,
        img_transform: Callable,
        txt_transform: Callable,
        dual_transform: Optional[Callable] = None,
        split: Optional[str] = None,
    ):
        self.config: Dict = yaml.safe_load(open(CONFIG_PATH))
        self.dataset_remote = self.config.get("dataset_remote", False)
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
        self.data = pd_read_parquet(
            dataset_path,
            self.dataset_remote,
            filters=filters,
        )

        self.index = self.data.index
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        self.dual_transform = dual_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        idx,
    ) -> Dict[str, Tensor]:
        if is_tensor(idx):
            idx = idx.tolist()
        idx = self.index[idx]
        img_path = self.data.loc[idx, "imgPath"]
        img = self.img_transform(
            self.get_img(
                local_path=img_path,
                remote_path=img_path,
                name=idx,
            )
        )
        txt = self.txt_transform(self.data.loc[idx, "txt"])
        if self.dual_transform is not None:
            txt, img = self.dual_transform([txt], img.unsqueeze(0))
        lat = self.data.loc[idx, "lat"]
        lon = self.data.loc[idx, "lon"]
        label = self.data.loc[idx, "price_fractional"]
        return {
            "id": idx,
            "img": img,
            "txt": txt,
            "coords": Tensor([lat, lon]),
            "label": Tensor([label]),
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
        path = str(PurePosixPath(IMAGES.local_path).joinpath(name).with_suffix(".jpg"))
        return Image.open(path)

    def iter_txt(self) -> Generator:
        for _, item in self.data["txt"].iteritems():
            yield item


class FoodPricingLazyDataset(Dataset):
    def __init__(
        self,
        img_transform,
        txt_transform,
        split: str = None,
    ) -> None:
        self.config: Dict = yaml.safe_load(open(CONFIG_PATH))
        self.dataset_remote = self.config.get("dataset_remote", False)
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
    ) -> Dict[str, Tensor]:
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
            "coords": Tensor([lat, lon]),
            "label": Tensor([label]),
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
        path = str(PurePosixPath(IMAGES.local_path).joinpath(name).with_suffix(".jpg"))
        return Image.open(path)

    def iter_txt(self) -> Generator:
        for _, item in self.data["txt"].iteritems():
            yield item
