from .external.location_coordinates import make_coordinates_table
from .internal.images import (
    CITIES_PATH,
    IMG_PATH_PATTERN,
    IMGS_SUBPATH,
    ZONES_SUBPATH,
    make_images_table,
)
from .internal.parquetize import (
    ITEM_DTYPES,
    ITEM_INPUT_PATH,
    ITEM_SUFFIX,
    csv_to_parquet,
)
from .table_model import Table

IMAGES_TABLE = Table(
    path=["interim"],
    file_name="images",
    file_format=".parquet.gzip",
    base_url_position=1,
    write_func=make_images_table,
    kwargs={
        "cities_path": CITIES_PATH,
        "zones_subpath": ZONES_SUBPATH,
        "imgs_subpath": IMGS_SUBPATH,
        "img_path_pattern": IMG_PATH_PATTERN,
        "get_shapes": False,
    },
    columns=[
        "imgPath",
        "store",
        "menuRow",
        "city",
        "zone",
    ],
    join_on=[
        "city",
        "zone",
        "store",
        "menuRow",
    ],
    categoricals=[
        "city",
        "zone",
    ],
)

COORDINATES_TABLE = Table(
    path=["external", "geopy"],
    file_name="coordinates",
    file_format=".parquet.gzip",
    base_url_position=1,
    write_func=make_coordinates_table,
    kwargs={},
    columns=[
        "city",
        "zone",
        "lat",
        "lon",
    ],
    join_on=[
        "city",
        "zone",
    ],
    categoricals=[
        "city",
        "zone",
    ],
)

ITEMS_TABLE = Table(
    path=["interim"],
    file_name="items",
    file_format=".parquet.gzip",
    base_url_position=1,
    write_func=csv_to_parquet,
    kwargs={
        "suffix": ITEM_SUFFIX,
        "dtypes": ITEM_DTYPES,
        "ipath": ITEM_INPUT_PATH,
    },
    columns=[
        "name",
        "description",
        "price_fractional",
        "menuRow",
        "city",
        "zone",
        "store",
    ],
    join_on=[
        "city",
        "zone",
        "store",
        "menuRow",
    ],
    categoricals=[
        "city",
        "zone",
    ],
)

FULL_TABLE = Table(
    path=["processed"],
    base_url_position=1,
    file_name="dataset",
    file_format=".parquet.gzip",
)
