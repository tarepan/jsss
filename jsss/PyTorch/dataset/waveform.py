"""
# Corpus/Dataset guide
Corpus: Distributed data
Dataset: Processed corpus for specific purpose
For example, JSUT corpus contains waves, and can be processed into JSUT-spec dataset which is made of spectrograms.
"""

# from typing import Callable, List, Literal, NamedTuple # >= Python3.8
from typing import Callable, List, NamedTuple, Optional
from pathlib import Path

from torch import Tensor, save, load
from torch.utils.data import Dataset
# currently there is no stub in torchaudio [issue](https://github.com/pytorch/audio/issues/615)
from torchaudio import load as load_wav
from torchaudio.transforms import Resample

from ...fs import try_to_acquire_archive_contents, save_archive, acquire_zip_fs
from ...corpus import ItemIdJSSS, Mode, JSSS


def get_dataset_wave_path(dir_dataset: Path, id: ItemIdJSSS) -> Path:
    return dir_dataset / id.mode / "waves" / f"{id.serial_num}.wave.pt"


def preprocess_as_wave(corpus: JSSS, dir_dataset: Path, new_sr: Optional[int] = None) -> None:
    """
    Transform JSSS corpus contents into waveform Tensor.
    Before this preprocessing, corpus contents should be deployed.
    """
    for id in corpus.get_identities():
        # todo: sr adjustment
        waveform, _sr_orig = load_wav(corpus.get_item_path(id))
        if new_sr is not None:
            waveform = Resample(_sr_orig, new_sr)(waveform)
        # :: [1, Length] -> [Length,]
        waveform: Tensor = waveform[0, :]
        path_wave = get_dataset_wave_path(dir_dataset, id)
        path_wave.parent.mkdir(parents=True, exist_ok=True)
        save(waveform, path_wave)


class Datum_JSSS_wave(NamedTuple):
    """
    Datum of JSSS dataset
    """

    waveform: Tensor
    label: str


class JSSS_wave(Dataset): # I failed to understand this error
    """
    Audio waveform dataset from JSSS speech corpus.
    This dataset yield (audio, label).
    """
    def __init__(
        self,
        modes: List[Mode] = ["short-form/basic5000"],
        download_corpus: bool = False,
        dir_data: str = "./data/",
        corpus_adress: Optional[str] = None,
        dataset_adress: str = "./data/datasets/JSSS_wave/archive/dataset.zip",
        resample_sr: Optional[int] = None,
        transform: Callable[[Tensor], Tensor] = (lambda i: i),
    ):
        """
        Args:
            mode: Sub corpus types.
            download_corpus: Whether download the corpus or not when dataset is not found.
            dir_data: Directory in which corpus and dataset are saved.
            corpus_adress: URL/localPath of corpus archive (remote url, like `s3::`, can be used). None use default URL.
            dataset_adress: URL/localPath of dataset archive (remote url, like `s3::`, can be used).
            resample_sr: If specified, resample with specified sampling rate.
            transform: Tensor transform on load.
        """
        # Design Notes:
        #   Dataset is often saved in the private adress, so there is no `download_dataset` safety flag.
        #   `download` is common option in torchAudio datasets.

        # Store parameters.
        self._resample_sr = resample_sr
        self._transform = transform
        self._dir_data = dir_data

        # Directory structure:
        # {dir_data}/
        #   corpuses/...
        #   datasets/
        #     JSSS_wave/
        #       archive/dataset.zip
        #       contents/{extracted dirs & files}
        self._corpus = JSSS(download_corpus, corpus_adress, f"{dir_data}/corpuses/JSSS/")
        self._path_archive_local = Path(dir_data)/"datasets"/"JSSS_wave"/"archive"/"dataset.zip"
        self._path_contents_local = Path(dir_data)/"datasets"/"JSSS_wave"/"contents"

        # Prepare the dataset.
        self._ids: List[ItemIdJSSS] = list(filter(lambda id: id.mode in modes, self._corpus.get_identities()))
        contents_acquired = try_to_acquire_archive_contents(
            self._path_contents_local,
            self._path_archive_local,
            dataset_adress,
            True
        )
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            # save dataset archive
            save_archive(self._path_contents_local, self._path_archive_local, dataset_adress)
            print("Dataset contents was generated and archive was saved.")

    def _generate_dataset_contents(self) -> None:
        """
        Generate dataset with corpus auto-download and preprocessing.
        """
        self._corpus.get_contents()
        preprocess_as_wave(self._corpus, self._path_contents_local, self._resample_sr)

    def _load_datum(self, id: ItemIdJSSS) -> Datum_JSSS_wave:
        waveform: Tensor = load(get_dataset_wave_path(self._path_contents_local, id))
        return Datum_JSSS_wave(self._transform(waveform), f"{id.mode}-{id.serial_num}")

    def __getitem__(self, n: int) -> Datum_JSSS_wave:
        """Load the n-th sample from the dataset.
        Args:
            n: The index of the datum to be loaded
        """
        return self._load_datum(self._ids[n])

    def __len__(self) -> int:
        return len(self._ids)


if __name__ == "__main__":
    pass
