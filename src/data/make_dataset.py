import shutil

import yaml
from src.data.config import DATASET, FULL_TABLE, IMAGES, TXT_TRAIN
from src.data.config_interim import COORDINATES_TABLE, IMAGES_TABLE, ITEMS_TABLE
from src.data.dataobj_model import DataObject
from src.data.storage import CONFIG_PATH, exists, get_S3_fs


def download(dataobj: DataObject) -> None:
    S3 = get_S3_fs()
    S3.download(str(dataobj.remote_path), str(dataobj.local_path), recursive=True)


def make_dataobj(dataobj: DataObject, remote: bool, **kwargs) -> None:
    opath = dataobj.remote_path if remote else dataobj.local_path
    dataobj.write_func(opath=opath, remote=dataobj.remote, **dataobj.kwargs, **kwargs)


def main(
    overwrite: bool = False,
    remote: bool = True,
    create_train_txt: bool = True,
    download_thumbnails: bool = True,
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
    else:
        if exists(COORDINATES_TABLE.remote_path, local=False):
            pass
        else:
            COORDINATES_TABLE.remote = remote
            make_dataobj(COORDINATES_TABLE, remote)

        if exists(IMAGES_TABLE.remote_path, local=False):
            pass
        else:
            IMAGES_TABLE.remote = remote
            make_dataobj(IMAGES_TABLE, remote)

        if exists(ITEMS_TABLE.remote_path, local=False):
            pass
        else:
            ITEMS_TABLE.remote = remote
            make_dataobj(ITEMS_TABLE, remote)

        if exists(FULL_TABLE.remote_path, local=False):
            pass
        else:
            make_dataobj(
                table=FULL_TABLE,
                remote=remote,
                tables=[ITEMS_TABLE, COORDINATES_TABLE, IMAGES_TABLE],
            )

        DATASET.remote = remote
        train_dev_test_ratio = (train_ratio, dev_ratio, test_ratio)
        make_dataobj(
            DATASET,
            remote,
            raw_table=FULL_TABLE,
            train_dev_test_ratio=train_dev_test_ratio,
            seed=seed,
        )

        if remote:
            download(DATASET)

        if create_train_txt:
            TXT_TRAIN.remote = remote
            make_dataobj(
                TXT_TRAIN,
                remote,
                raw_table=DATASET,
            )

        if remote:
            download(TXT_TRAIN)

        if download_thumbnails:
            make_dataobj(
                IMAGES,
                remote=False,  # thumbnails are always downloaded locally
                raw_table=DATASET,
            )

        config = {
            "dataset_remote": False,
            "txt_created": create_train_txt,
            "img_thumbnails": download_thumbnails,
        }
        with open(CONFIG_PATH, mode="w") as f:
            yaml.dump(config, f)


if __name__ == "__main__":
    main(overwrite=False, remote=True)
