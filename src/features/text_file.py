from pathlib import PurePosixPath
from typing import Any, List, Tuple, Union

from ..data.storage import dd_read_parquet, get_S3_fs
from ..data.dataobj_model import Table


def create_txt(
    opath: PurePosixPath,
    remote: bool = False,
    raw_table: Table = Table(),
    columns: Union[List[str], None] = None,
    filters: Union[List[List[Tuple[str, str, Any]]], None] = None,
) -> None:
    path = raw_table.remote_path if raw_table.remote else raw_table.local_path
    ddf = dd_read_parquet(
        path,
        raw_table.remote,
        columns,
        filters=filters,
    )
    if (oformat := opath.suffix) != ".txt":
        raise Exception(f"Unsupported text file format {oformat}. Use .txt")
    if remote:
        S3 = get_S3_fs()
        f = S3.open(opath, mode="w")
    else:
        f = open(opath, mode="w")
    try:
        for _, row in ddf.iterrows():
            for col in columns:
                line = row[col]
                if not isinstance(line, str):
                    raise Exception(
                        f"Line: {line} in row: {row.name}, col: {col} is not a string."
                    )
                f.write(line + "\n")
        f.close()
    except Exception as e:
        f.close()
        raise e
