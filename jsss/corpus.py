from jsss.getGoogleDriveContents import getGDriveLargeContents
from typing import Dict, List, NamedTuple, Optional
from pathlib import Path

import fsspec # type: ignore
from fsspec.utils import get_protocol # type: ignore

from .fs import try_to_acquire_archive_contents


# ## Glossary
# - archive: Single archive file.
# - contents: A directory in which archive's contents exist.


# summarization have an issue about complicated file name, so currently not supported.  


# Shortform = Literal["short-form/basic5000", "short-form/onomatopee300", "short-form/voiceactress100"] # >=Python3.8
# Longform = Literal["long-form/katsura-masakazu", "long-form/udon", "long-form/washington-dc"] # >=Python3.8
# Mode = Literal[Longform, Shortform, "simplification", "summarization"] # >=Python3.8
Mode = str
modes = [
    "short-form/basic5000",
    "short-form/onomatopee300",
    "short-form/voiceactress100",
    "long-form/katsura-masakazu",
    "long-form/udon",
    "long-form/washington-dc",
    "simplification",
    "summarization"
]


class ItemIdJSSS(NamedTuple):
    mode: Mode
    serial_num: str


class JSSS:
    def __init__(
        self,
        download : bool = False,
        url: Optional[str] = None,
        dir_corpus_local: str = "./data/corpuses/jsss/"
    ) -> None:
        """
        Wrapper of JSSS corpus.
        [Official Website](https://sites.google.com/site/shinnosuketakamichi/research-topics/jsss_corpus).
        Corpus will be deployed as below.

        {dir_corpus_local}/
            archive/
                f"{corpus_name}.zip"
            contents/
                {extracted dirs & files}

        Args:
            download: Download corpus when there is no archive in local.
            url: Corpus archive url (Various url type (e.g. S3, GCP) is accepted through `fsspec` library).
            dir_corpus_local: Corpus directory.
        """
        ver: str = "ver1"
        # Equal to 1st layer directory name of original zip.
        self._corpus_name: str = f"jsss_{ver}"

        self._gdrive_id: str = "1NyiZCXkYTdYBNtD1B-IMAYCVa-0SQsKX"
        self._url = url
        self._download = download
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem(get_protocol(self._url if self._url else "./"))

        self._path_archive_local = Path(dir_corpus_local) / "archive" / f"{self._corpus_name}.zip"
        self._path_contents_local = Path(dir_corpus_local) / "contents"

    def get_archive(self) -> None:
        """
        Get the corpus archive file.
        """
        # library selection:
        #   `torchaudio.datasets.utils.download_url` is good for basic purpose, but not compatible with private storages.
        # todo: caching
        path_archive = self._path_archive_local
        if path_archive.exists():
            if path_archive.is_file():
                print("Archive file already exists.")
            else:
                raise RuntimeError(f"{str(path_archive)} should be archive file or empty, but it is directory.")
        else:
            if self._download:
                if self._url:
                    self._fs.get_file(self._url, path_archive)
                else:
                    # from original Google Drive
                    size_GB = 1.01
                    getGDriveLargeContents(self._gdrive_id, path_archive, size_GB)
            else:
                raise RuntimeError("Try to get_archive, but `download` is disabled.")

    def get_contents(self) -> None:
        """
        Get the archive and extract the contents if needed.
        """
        # todo: caching
        path_contents = self._path_contents_local

        acquired = try_to_acquire_archive_contents(
            path_contents,
            self._path_archive_local,
            self._url,
            self._download,
            self._gdrive_id,
            1.01)
        if not acquired:
            raise RuntimeError(f"Specified corpus archive cannot be acquired. Check the link (`{self._url}`) or `download` option.")

    def get_identities(self) -> List[ItemIdJSSS]:
        """
        Get corpus item identities.

        Returns:
            Full item identity list.
        """
        divs = {
            "short-form/basic5000": map(lambda num: str(num).zfill(4), range(1, 3001)),
            "short-form/onomatopee300": map(lambda num: str(num).zfill(3), range(1, 186)),
            "short-form/voiceactress100": map(lambda num: str(num).zfill(3), range(1, 101)),
            "long-form/katsura-masakazu": map(lambda num: str(num).zfill(2), range(1, 60)),
            "long-form/udon": map(lambda num: str(num).zfill(2), range(1, 87)),
            "long-form/washington-dc": map(lambda num: str(num).zfill(2), range(1, 24)),
            "simplification": map(lambda num: str(num).zfill(3), range(1, 228)),
            "summarization": map(lambda num: str(num).zfill(4), range(1, 227))
        }
        ids: List[ItemIdJSSS] = []
        for mode in modes:
                for num in divs[mode]:
                    ids.append(ItemIdJSSS(mode, num))
        return ids

    def get_item_path(self, id: ItemIdJSSS) -> Path:
        """
        Get path of the item.

        Args:
            id: Target item identity.
        Returns:
            Path of the specified item.
        """
        prefix: Dict[Mode, str] = {
            "short-form/basic5000": "BASIC5000",
            "short-form/onomatopee300": "ONOMATOPEE300",
            "short-form/voiceactress100": "VOICEACTRESS100",
            "long-form/katsura-masakazu": "KATSURA-MASAKAZU",
            "long-form/udon": "UDON",
            "long-form/washington-dc": "WASHINGTON-DC",
            "simplification": "SIMPLIFICATION",
            "summarization": "SUMMARIZATION"
        }
        p = f"{str(self._path_contents_local)}/{self._corpus_name}/{id.mode}/wav24kHz16bit/{prefix[id.mode]}_{id.serial_num}.wav"
        return Path(p)
