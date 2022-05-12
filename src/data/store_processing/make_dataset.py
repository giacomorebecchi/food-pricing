from src.data.storage import (
    build_output_path,
    build_path,
    get_S3_fs,
    load_list,
    write_list,
)
from src.data.store_processing.parsing import transform_data_menu, transform_data_store

GLOB_PATH = build_path(
    "data",
    "content",
    "*",
    "store",
    "*",  # city
    "*",  # zone
    "*",  # job_ID
    "*",  # timestamp
    "*",  # file_name
)
PATH_MAP_STORE = {
    "/data/": "/data-interim/",
    "/content/": "/processed/",
    ".jsonlines": "-store.csv",
}
PATH_MAP_MENU = {
    "/data/": "/data-interim/",
    "/content/": "/processed/",
    ".jsonlines": "-menu.csv",
}


def make_dataset() -> None:
    S3 = get_S3_fs()
    for file_path in S3.glob(GLOB_PATH):
        content = load_list(fs=S3, path=file_path)
        store_data = transform_data_store(content)
        output_path_store = build_output_path(file_path, PATH_MAP_STORE)
        write_list(S3, output_path_store, store_data)
        menu_data = transform_data_menu(content)
        output_path_menu = build_output_path(file_path, PATH_MAP_MENU)
        write_list(S3, output_path_menu, menu_data)


if __name__ == "__main__":
    make_dataset()
