import pandas as pd
from src.data.location_list.parsing import find_locations, get_sitemap
from src.data.storage import get_local_data_path


def make_dataset() -> None:
    sitemap_text = get_sitemap()
    locations = find_locations(sitemap_text)
    df = pd.DataFrame(locations, columns=["city", "zone"])
    path = get_local_data_path(
        ["external", "location"],
        file_name="city_zone",
        file_format=".parquet",
    )
    df.to_parquet(path, compression="gzip")


if __name__ == "__main__":
    make_dataset()
