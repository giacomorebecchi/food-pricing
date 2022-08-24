from pathlib import PurePosixPath

import pandas as pd
from src.data.external.geocode_location import geocode_address, get_coords
from src.data.external.utils import find_locations, get_sitemap
from src.data.storage import pd_write_parquet


def make_coordinates_table(opath: PurePosixPath, remote: bool = False) -> None:
    sitemap_text = get_sitemap()
    locations = find_locations(sitemap_text)
    df = pd.DataFrame(locations, columns=["city", "zone"])
    coords = pd.DataFrame(index=df.index)
    coords["lat"], coords["lon"] = zip(
        *df.apply(lambda x: get_coords(x["city"], x["zone"]), axis=1)
    )
    geocode_address.store_cache()
    output = df.join(coords)
    pd_write_parquet(opath, output, remote)


if __name__ == "__main__":
    make_coordinates_table()
