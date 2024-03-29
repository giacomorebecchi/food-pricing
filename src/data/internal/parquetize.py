import os
from pathlib import PurePosixPath
from typing import Dict

import dask.dataframe as dd
import pandas as pd
from dask import delayed
from dotenv import load_dotenv
from src.data.storage import (
    dd_write_parquet,
    get_local_data_path,
    get_remote_data_path,
    get_S3_fs,
)

load_dotenv()

ITEM_SUFFIX = "menu"
STORE_SUFFIX = "store"

input_path = lambda s: get_remote_data_path(
    path=["processed", "store", "**"],
    file_name=f"*-{s}",
    file_format=".csv",
    base_url_position=1,
)
ITEM_INPUT_PATH = input_path(ITEM_SUFFIX)
STORE_INPUT_PATH = input_path(STORE_SUFFIX)

output_path = lambda s: get_local_data_path(
    path=["interim"], file_name=s, file_format=""
)
ITEM_OUTPUT_PATH = output_path(ITEM_SUFFIX)
STORE_OUTPUT_PATH = output_path(STORE_SUFFIX)

ITEM_DTYPES = {
    "percentageDiscounted": str,
    "name": object,
    "popular": bool,
    "categoryId": int,
    "modifierGroupIds": object,
    "priceDiscounted": str,
    "isSignatureExclusive": bool,
    "productInformation": str,
    "nutritionalInfo": str,
    "available": bool,
    "maxSelection": str,
    "description": object,
    "id": int,
    "priceDiscounted_code": str,
    "priceDiscounted_fractional": float,
    "priceDiscounted_formatted": str,
    "price_code": str,
    "price_fractional": float,
    "price_formatted": str,
    "image_typeName": str,
    "image_altText": str,
    "image_url": str,
    "image_type": str,
    "image": str,
}
STORE_DTYPES = {
    "id": int,
    "name": str,
    "hasOrderNotes": bool,
    "tipMessage": str,
    "menuDisabled": bool,
    "deliversToCustomerLocation": bool,
    "menuId": int,
    "drnId": str,
    "currencyCode": str,
    "currencySymbol": str,
    "location_cityId": int,
    "location_zoneId": int,
    "location_address_address1": str,
    "location_address_postCode": str,
    "location_address_neighborhood": str,
    "location_address_city": str,
    "location_address_country": str,
}


@delayed
def load_csv(fp: str, suffix: str, dtypes=Dict[str, type]) -> pd.DataFrame:
    S3fp = "s3://" + fp
    raw_df = pd.read_csv(
        S3fp,
        storage_options={"client_kwargs": {"endpoint_url": os.environ["S3_ENDPOINT"]}},
    )
    df = pd.DataFrame(
        raw_df,
        columns=list(dtypes.keys()),
    ).astype(dtypes)
    df["filePath"] = fp
    fp_components = fp.split("/")
    df["city"] = fp_components[5]
    df["zone"] = fp_components[6]
    df["store"] = fp_components[9].removesuffix(f"-{suffix}.csv")
    df["menuRow"] = df.index
    return df


def csv_to_parquet(
    suffix: str,
    dtypes: Dict[str, type],
    ipath: str,
    opath: PurePosixPath,
    remote: bool = False,
) -> None:
    S3 = get_S3_fs()
    csv_paths = S3.glob(str(ipath))
    delayed_dfs = (load_csv(fp, suffix, dtypes) for fp in csv_paths)
    ddf: dd.DataFrame = dd.from_delayed(
        delayed_dfs,
        meta={
            **dtypes,
            "filePath": str,
            "city": str,
            "zone": str,
            "store": str,
            "menuRow": int,
        },
    )
    dd_write_parquet(opath, ddf, remote, partition_on=["city", "zone"])


def make_items_table() -> None:
    csv_to_parquet(ITEM_SUFFIX, ITEM_DTYPES, ITEM_INPUT_PATH, ITEM_OUTPUT_PATH)


def make_stores_table() -> None:
    csv_to_parquet(STORE_SUFFIX, STORE_DTYPES, STORE_INPUT_PATH, STORE_OUTPUT_PATH)


if __name__ == "__main__":
    make_items_table()
    make_stores_table()
