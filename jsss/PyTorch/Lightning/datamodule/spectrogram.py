from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
from torch.tensor import Tensor
from torch.utils.data import random_split, DataLoader

from ...dataset.spectrogram import JSSS_spec
from ....corpus import Mode


class JSSS_spec_DataModule(pl.LightningDataModule):
    """
    JSSS_spec dataset's PyTorch Lightning datamodule
    """

    def __init__(
        self,
        batch_size: int,
        download: bool,
        dir_root: str = "./data/",
        modes: List[Mode] = ["short-form/basic5000"],
        transform: Callable[[Tensor], Tensor] = lambda i: i,
        corpus_adress: Optional[str] = None,
        dataset_adress: str = "./data/datasets/JSSS_spec/archive/dataset.zip",
    ):
        super().__init__()
        self.n_batch = batch_size
        self.download = download
        self.dir_root = dir_root
        self.modes = modes
        self.transform = transform
        self.corpus_adress = corpus_adress
        self.dataset_adress = dataset_adress

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, stage: Union[str, None] = None) -> None:
        if stage == "fit" or stage is None:
            dataset_train = JSSS_spec(
                modes=self.modes,
                transform=self.transform,
                download_corpus=self.download,
                dir_data=self.dir_root,
                corpus_adress=self.corpus_adress,
                dataset_adress=self.dataset_adress,
            )
            n_train = len(dataset_train)
            self.data_train, self.data_val = random_split(
                dataset_train, [n_train - 10, 10]
            )
        if stage == "test" or stage is None:
            self.data_test = JSSS_spec(
                modes=self.modes,
                transform=self.transform,
                download_corpus=self.download,
                dir_data=self.dir_root,
                corpus_adress=self.corpus_adress,
                dataset_adress=self.dataset_adress,
            )

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.data_train, batch_size=self.n_batch)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.data_val, batch_size=self.n_batch)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.data_test, batch_size=self.n_batch)


if __name__ == "__main__":
    print("This is datamodule/waveform.py")
    # If you use batch (n>1), transform function for Tensor shape rearrangement is needed
    dm_JSSS_spec = JSSS_spec_DataModule(1, download=True)

    # download & preprocessing
    dm_JSSS_spec.prepare_data()

    # runtime setup
    dm_JSSS_spec.setup(stage="fit")

    # yield dataloader
    dl = dm_JSSS_spec.train_dataloader()
    print(next(iter(dl)))
    print("datamodule/waveform.py test passed")