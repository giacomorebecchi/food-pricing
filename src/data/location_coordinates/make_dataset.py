import pandas as pd
from src.data.location_coordinates.parsing import geocode_address, get_coords
from src.data.storage import get_local_data_path


def make_dataset() -> None:
    input_path = get_local_data_path(
        ["external", "location"],
        file_name="city_zone",
        file_format=".parquet",
    )
    df = pd.read_parquet(input_path)
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


if __name__ == "__main__":
    make_dataset()
