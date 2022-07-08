import shutil

import yaml
from src.data.config import DATASET, FULL_TABLE  # TODO: import TXT_TRAIN
from src.data.config_interim import COORDINATES_TABLE, IMAGES_TABLE, ITEMS_TABLE
from src.data.storage import CONFIG_PATH, exists, get_S3_fs
from src.data.table_model import Table


def download(table: Table) -> None:
    S3 = get_S3_fs()
    S3.download(str(table.remote_path), str(table.local_path), recursive=True)


def make_table(table: Table, remote: bool, **kwargs) -> None:
    opath = table.remote_path if remote else table.local_path
    table.write_func(opath=opath, remote=table.remote, **table.kwargs, **kwargs)


def main(
    overwrite: bool = False,
    remote: bool = True,
    # TODO: insert bool argument for creating txt
    train_ratio: float = 0.7,
    dev_ratio: float = None,
    test_ratio: float = None,
    seed: int = 42,
) -> None:
    if exists(DATASET.local_path, local=True):
        if overwrite:
            # delete the table
            try:
                shutil.rmtree(DATASET.local_path)
                # TODO: delete the txt file
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        else:
            # TODO: check that the partition ratios are respected, else
            # load the table, overwrite the partition and overwrite the table
            raise Exception("Data has already been downloaded.")
    if exists(DATASET.remote_path, local=False) and remote and not overwrite:
        download(DATASET)
        # TODO: check that the partition ratios are respected, else
        # load the table, overwrite the partition and overwrite the table
    # TODO: if exists txt file, download
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
        train_dev_test_ratio = (train_ratio, dev_ratio, test_ratio)
        make_table(
            DATASET,
            remote,
            raw_table=FULL_TABLE,
            train_dev_test_ratio=train_dev_test_ratio,
            seed=seed,
        )
        # TODO: create txt file and store it

        if remote:
            download(DATASET)

        # TODO: uncomment this and add txt_remote and txt_created
        # # TODO: if in the future there will be the need to do so,
        # # write to the config.yml file with this:
        # config = {"dataset_remote": False, "image_remote": True}
        # with open(CONFIG_PATH, mode="w") as f:
        #     yaml.dump(config, f)


if __name__ == "__main__":
    main(overwrite=False, remote=True)
