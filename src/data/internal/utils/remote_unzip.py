import logging
import zipfile
from pathlib import PurePosixPath

from src.data.settings import get_S3_settings
from src.data.storage import get_S3_fs
from src.definitions import ROOT_DIR

ZIP_PATH = PurePosixPath(ROOT_DIR).joinpath("scp")

logging.basicConfig(level=logging.INFO)


def main():
    FILE_FORMATS = set(
        (
            ".json",
            ".txt",
            ".xml",
            ".parquet",
            ".jpeg",
            ".jpg",
            ".csv",
            ".jsonlines",
            ".gzip",
            ".html",
        )
    )
    S3 = get_S3_fs()
    with open(ZIP_PATH, mode="rb") as zipf:
        z = zipfile.ZipFile(zipf)
        for filename in z.namelist():
            path = PurePosixPath(get_S3_settings().BUCKET).joinpath(
                *PurePosixPath(filename).parts[4:]
            )
            if path.suffix in FILE_FORMATS:
                logging.info(f"Filename: {filename}")
                logging.info(f"Path: {path}")
                with S3.open(str(path), mode="wb") as f:
                    f.write(z.read(filename))
                    logging.info(f"Decompressed file {path}")


if __name__ == "__main__":
    main()
