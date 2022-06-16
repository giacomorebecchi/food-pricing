import shutil

from src.data.config import DATASET, FULL_TABLE
from src.data.config_interim import COORDINATES_TABLE, IMAGES_TABLE, ITEMS_TABLE
from src.data.storage import exists, get_S3_fs
from src.data.table_model import Table


def download(table: Table) -> None:
    S3 = get_S3_fs()
    S3.download(str(table.remote_path), str(table.local_path), recursive=True)


def make_table(table: Table, remote: bool, **kwargs) -> None:
    opath = table.remote_path if remote else table.local_path
    table.write_func(opath=opath, remote=table.remote, **table.kwargs, **kwargs)


def main(overwrite: bool = False, remote: bool = True) -> None:
    if exists(DATASET.local_path, local=True):
        if overwrite:
            try:
                shutil.rmtree(DATASET.local_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        else:
            raise Exception("Data has already been downloaded.")
    if exists(DATASET.remote_path, local=False):
        download(DATASET)
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

        if exists(FULL_TABLE.remote_path, local=False):
            pass
        else:
            make_table(
                table=FULL_TABLE,
                remote=remote,
                tables=[ITEMS_TABLE, COORDINATES_TABLE, IMAGES_TABLE],
            )

        DATASET.remote = remote
        make_table(DATASET)

        if remote:
            download(DATASET)


if __name__ == "__main__":
    main(overwrite=False, remote=True)
