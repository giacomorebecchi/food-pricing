import pandas as pd
from src.data.storage import get_local_data_path

from .geocode_location import geocode_address, get_coords
from .utils import find_locations, get_sitemap


def make_coordinates_table() -> None:
    sitemap_text = get_sitemap()
    locations = find_locations(sitemap_text)
    df = pd.DataFrame(locations, columns=["city", "zone"])
    path = get_local_data_path(
        ["external", "location"],
        file_name="city_zone",
        file_format=".parquet",
    )
    coords = pd.DataFrame(index=df.index)
    coords["lat"], coords["lon"] = zip(
        *df.apply(lambda x: get_coords(x["city"], x["zone"]), axis=1)
    )
    geocode_address.store_cache()
    output_path = get_local_data_path(
        ["external", "geopy"],
        file_name="coordinates",
        file_format=".parquet",
    )
    output = df.join(coords)
    output.to_parquet(output_path, compression="gzip")
    df.to_parquet(path, compression="gzip")


if __name__ == "__main__":
    make_coordinates_table()
