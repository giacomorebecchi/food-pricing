from src.data.location_content.parsing import parse_location_response
from src.data.storage import (
    build_output_path,
    build_path,
    get_S3_fs,
    load_bytes,
    write_bytes,
)

DATA_XPATH = "//script[contains(@id, '__NEXT_DATA__')]"
GLOB_PATH = build_path(
    "data",
    "response",
    "*",
    "location",
    "*",  # city
    "*",  # zone
    "*",  # job_ID
    "*",  # file_name
)
PATH_MAP = {
    "/data/": "/data-interim/",
    "/response/": "/content/",
    ".html": ".json",
}


def make_dataset() -> None:
    S3 = get_S3_fs()
    for file_path in S3.glob(GLOB_PATH):
        response_content = load_bytes(fs=S3, path=file_path)
        data = parse_location_response(response_content, DATA_XPATH)
        output_path = build_output_path(file_path, PATH_MAP)
        write_bytes(S3, output_path, data)


if __name__ == "__main__":
    make_dataset()
