import shutil

from src.data.config import (
    COORDINATES_TABLE,
    FULL_TABLE,
    IMAGES_TABLE,
    ITEMS_TABLE,
    Table,
)
from src.data.join_tables import join
from src.data.storage import exists, get_S3_fs


def download(table: Table) -> None:
    S3 = get_S3_fs()
    S3.download(table.remote_path, table.local_path, recursive=True)


def make_table(table: Table, remote: bool) -> None:
    opath = table.remote_path if remote else table.local_path
    table.write_func(opath=opath, remote=table.remote, **table.kwargs)


def main(overwrite: bool = False, remote: bool = True) -> None:
    if exists(FULL_TABLE.local_path, local=True):
        if overwrite:
            try:
                shutil.rmtree(FULL_TABLE.local_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        else:
            raise Exception("Data has already been downloaded.")
    if exists(FULL_TABLE.remote_path, local=False):
        download(FULL_TABLE)
    else:
        if exists(COORDINATES_TABLE.remote_path, local=False):
            pass
        else:
            COORDINATES_TABLE.remote = remote
            make_table(COORDINATES_TABLE, remote)

        if exists(IMAGES_TABLE.remote_path, local=False):
            pass
        else:
            IMAGES_TABLE.remote = remote
            make_table(IMAGES_TABLE, remote)

        if exists(ITEMS_TABLE.remote_path, local=False):
            pass
        else:
            ITEMS_TABLE.remote = remote
            make_table(ITEMS_TABLE, remote)

        opath = FULL_TABLE.remote_path if remote else FULL_TABLE.local_path
        join([ITEMS_TABLE, COORDINATES_TABLE, IMAGES_TABLE], opath=opath, remote=remote)


if __name__ == "__main__":
    main(overwrite=False, remote=False)
