from typing import Optional, Union, NamedTuple, Dict, List
from pathlib import Path

# import fsspec # type: ignore
# from fsspec.utils import get_protocol # type: ignore

from .abstract_corpus import AbstractCorpus, get_contents
from .getGoogleDriveContents import forward_file_from_GDrive


# ## Glossary
# - archive: Single archive file.
# - contents: A directory in which archive's contents exist.


# summarization have an issue about complicated file name, so currently not supported.  


# Shortform = Literal["short-form/basic5000", "short-form/onomatopee300", "short-form/voiceactress100"] # >=Python3.8
# Longform = Literal["long-form/katsura-masakazu", "long-form/udon", "long-form/washington-dc"] # >=Python3.8
# Subtype = Literal[Longform, Shortform, "simplification", "summarization"] # >=Python3.8
Subtype = str
subtypes = [
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
    mode: Subtype
    serial_num: int


class JSSS(AbstractCorpus[ItemIdJSSS]):
    """JSSS corpus.
    
    Archive/contents handler of JSSS corpus.
    """

    gdrive_contents_id: str = "1NyiZCXkYTdYBNtD1B-IMAYCVa-0SQsKX"

    def __init__(
        self,
        adress: Optional[str] = None,
        download_origin : bool = False,
    ) -> None:
        """Initiate JSSS with archive options.

        Args:
            adress: Corpus archive adress (e.g. path, S3) from/to which archive will be read/written through `fsspec`.
            download_origin: Download original corpus when there is no corpus in local and specified adress.
        """

        ver: str = "ver1"
        # Equal to 1st layer directory name of original zip.
        self._corpus_name: str = f"jsss_{ver}"

        dir_corpus_local: str = "./data/corpuses/JSSS/"
        default_path_archive = str((Path(dir_corpus_local) / "archive" / f"{self._corpus_name}.zip").resolve())
        self._path_contents_local = Path(dir_corpus_local) / "contents"
        self._adress = adress if adress else default_path_archive

        self._download_origin = download_origin
        # self._fs: fsspec.AbstractFileSystem = fsspec.filesystem(get_protocol(self._adress))

    # def get_archive(self) -> None:
    #     """
    #     Get the corpus archive file.
    #     """
    #     # library selection:
    #     #   `torchaudio.datasets.utils.download_url` is good for basic purpose, but not compatible with private storages.
    #     path_archive = self._path_archive_local
    #     if path_archive.exists():
    #         if path_archive.is_file():
    #             print("Archive file already exists.")
    #         else:
    #             raise RuntimeError(f"{str(path_archive)} should be archive file or empty, but it is directory.")
    #     else:
    #         if self._fs.exists(self._adress):
    #             self._fs.get_file(self._adress, path_archive)
    #         elif self._download_origin:
    #             # from original Google Drive
    #             size_GB = 1.01
    #             getGDriveLargeContents(self._gdrive_id, path_archive, size_GB)
    #         else:
    #             raise RuntimeError("Try to get an archive, but no file in the adress and `download_origin` is disabled.")

    def get_contents(self) -> None:
        """Get corpus contents into local.
        """

        get_contents(self._adress, self._path_contents_local, self._download_origin, self.forward_from_origin)

    def forward_from_origin(self) -> None:
        """Forward original corpus archive to the adress.
        """

        forward_file_from_GDrive(self.gdrive_contents_id, self._adress, 1.01)

    def get_identities(self) -> List[ItemIdJSSS]:
        """
        Get corpus item identities.

        Returns:
            Full item identity list.
        """
        divs = {
            "short-form/basic5000": range(1, 3001),
            "short-form/onomatopee300": range(1, 186),
            "short-form/voiceactress100": range(1, 101),
            "long-form/katsura-masakazu": range(1, 60),
            "long-form/udon": range(1, 87),
            "long-form/washington-dc": range(1, 24),
            "simplification": range(1, 228),
            "summarization": range(1, 227),
        }
        ids: List[ItemIdJSSS] = []
        for subtype in subtypes:
                for num in divs[subtype]:
                    ids.append(ItemIdJSSS(subtype, num))
        return ids

    def get_item_path(self, id: ItemIdJSSS) -> Path:
        """
        Get path of the item.

        Args:
            id: Target item identity.
        Returns:
            Path of the specified item.
        """
        name: Dict[Subtype, Dict[str, Union[str, int]]] = {
            "short-form/basic5000": {
                "prefix": "BASIC5000",
                "zpad": 4
            },
            "short-form/onomatopee300": {
                "prefix": "ONOMATOPEE300",
                "zpad": 3
            },
            "short-form/voiceactress100": {
                "prefix": "VOICEACTRESS100",
                "zpad": 3
            },
            "long-form/katsura-masakazu": {
                "prefix": "KATSURA-MASAKAZU",
                "zpad": 2
            },
            "long-form/udon": {
                "prefix": "UDON",
                "zpad": 2
            },
            "long-form/washington-dc": {
                "prefix": "WASHINGTON-DC",
                "zpad": 2
            },
            "simplification": {
                "prefix": "SIMPLIFICATION",
                "zpad": 3
            },
            "summarization": {
                "prefix": "SUMMARIZATION",
                "zpad": 3
            }
        }
        root = str(self._path_contents_local)
        prefix = name[id.mode]["prefix"]
        num = str(id.serial_num).zfill(int(name[id.mode]["zpad"]))
        p = f"{root}/{self._corpus_name}/{id.mode}/wav24kHz16bit/{prefix}_{num}.wav"
        return Path(p)
