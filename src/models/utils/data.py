from typing import Dict

import dask.dataframe as dd
from PIL import Image
from torch import is_tensor
from torch.utils.data import Dataset


class FoodPricingDataset(Dataset):
    def __init__(
        self,
        img_transform,
        txt_transform,
    ) -> None:
        self.remote_data = False  # TODO: Understand if dataset is saved or from S3
        self.remote_img = False  # TODO: Understand if images are saved or from S3
        self.data = dd.DataFrame()  # TODO: Load dataset
        self.img_folder = None  # TODO: Define image folder
        self.img_transform = img_transform
        self.txt_transform = txt_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        idx,  # TODO: Understand idx type
    ) -> Dict[str]:
        if is_tensor(idx):
            idx = idx.tolist()
        item_id = self.data.loc[idx, "id"]  # TODO: more efficient with index
        img = Image.open(self.data.loc[idx, "imgPath"]).convert("RGB")
        txt = self.data.loc[idx, "txt"]
        coords = (self.data.loc[idx, "lat"], self.data.loc[idx, "lon"])
        label = self.data.loc[idx, "price_fractional"]
        return {
            "id": item_id,  # TODO: Tensor
            "img": self.img_transform(img),
            "txt": self.txt_transform(txt),
            "coords": coords,  # TODO: Tensor
            "label": label,  # TODO: Tensor
        }
