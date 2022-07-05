from io import BytesIO
from typing import Dict, Generator, List

import torch
import yaml
from PIL import Image
from src.data.config import DATASET
from src.data.storage import CONFIG_PATH, dd_read_parquet, get_S3_fs
from torch import is_tensor
from torch.utils.data import Dataset


class FoodPricingDataset(Dataset):
    def __init__(
        self,
        img_transform,
        txt_transform,
    ) -> None:
        self.config: Dict = yaml.safe_load(open(CONFIG_PATH))
        self.dataset_remote = self.config.get("dataset_remote")
        self.img_remote = self.config.get("img_remote")
        dataset_path = (
            DATASET.remote_path if self.dataset_remote else DATASET.local_path
        )
        self.data = dd_read_parquet(dataset_path, self.dataset_remote)
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
        img = self.getimage(ddf.loc[idx, "imgPath"])
        txt = ddf.loc[idx, "txt"]
        lat = ddf.loc[idx, "lat"]
        lon = ddf.loc[idx, "lon"]
        label = ddf.loc[idx, "price_fractional"]
        return {
            "img": self.img_transform(img),
            "txt": self.txt_transform(txt),
            "coords": torch.Tensor([lat, lon]),
            "label": torch.IntTensor([label]),
        }

    def getimage(self, path: str) -> Image:
        if self.img_remote:
            S3 = get_S3_fs()
            with S3.open(path, mode="rb") as f:
                img_bytes = f.read()
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        else:
            return Image.open(path).convert("RGB")

    def iter_txt(self, idxs: List[int]) -> Generator:
        for idx in idxs:
            idx = self.index[idx]
            yield self.data.loc[idx, "txt"].compute().values[0]

    def set_txt_transform(self, txt_transform) -> None:
        self.txt_transform = txt_transform

    def set_img_transform(self, img_transform) -> None:
        self.img_transform = img_transform
