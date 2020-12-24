from jsss.getGoogleDriveContents import getGDriveLargeContents
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import fsspec
from fsspec.utils import get_protocol
from torchaudio.datasets.utils import extract_archive


def try_to_acquire_archive_contents(pull_from: str, extract_to: Path) -> bool:
    """
    Try to acquire the contents of the archive.
    Priority:
      1. (already extracted) local contents
      2. adress-specified (local|remote) archive through fsspec

    Returns:
        True if success_acquisition else False
    """
    pull_from_with_cache = f"simplecache::{pull_from}"

    # validation
    if extract_to.is_file():
        raise RuntimeError(f"contents ({str(extract_to)}) should be directory or empty, but it is file.")

    if extract_to.exists():
        # contents directory already exists.
        return True
    else:
        fs: fsspec.AbstractFileSystem = fsspec.filesystem(get_protocol(pull_from_with_cache))
        archiveExists = fs.exists(pull_from_with_cache)
        archiveIsFile = fs.isfile(pull_from_with_cache)

        # validation
        if archiveExists and (not archiveIsFile):
            raise RuntimeError(f"Archive ({pull_from_with_cache}) should be file or empty, but it is directory.")

        # A dataset file exist, so pull and extract.
        if archiveExists:
            extract_to.mkdir(parents=True, exist_ok=True)
            with fsspec.open(pull_from_with_cache, "rb") as archive:
                with NamedTemporaryFile("wb") as tmp:
                    tmp.write(archive.read())
                    tmp.seek(0)
                    extract_archive(tmp.name, str(extract_to))
            return True
        # no corresponding archive. Failed to acquire.
        else:
            return False


def save_archive(path_contents: Path, path_archive_local: Path, adress_archive: str) -> None:
    """
    Save contents as ZIP archive.

    Args:
        path_contents: Contents root directory path
        path_archive_local: Local path of newly generated archive file
        adress_archive: Saved adress
        compression: With-compression if True else no-compression
    """
    shutil.make_archive(str(path_archive_local.with_suffix("")), "zip", root_dir=path_contents)

    # write (==upload) the archive
    with open(path_archive_local, mode="rb") as stream_zip:
        with fsspec.open(f"simplecache::{adress_archive}", "wb") as f:
            f.write(stream_zip.read())
