from typing import Callable, List, Optional, Union

# currently there is no stub in pytorch lightning
import pytorch_lightning as pl  # type: ignore
from torch.tensor import Tensor
from torch.utils.data import random_split, DataLoader

from ...dataset.waveform import JSSS_wave
from ....corpus import Mode


class NpVCC2016DataModule(pl.LightningDataModule):
    """
    npVCC2016 speech corpus's PyTorch Lightning datamodule
    """

    def __init__(
        self,
        batch_size: int,
        download: bool,
        dir_root: str = "./data/",
        modes: List[Mode] = ["short-form/basic5000"],
        resample_sr: Optional[int] = None,
        transform: Callable[[Tensor], Tensor] = lambda i: i,
    ):
        super().__init__()
        self.n_batch = batch_size
        self.download = download
        self.dir_root = dir_root
        self.modes = modes
        self.transform = transform
        self._resample_sr = resample_sr
        # transforms.Compose([transforms.ToTensor()])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, stage: Union[str, None] = None) -> None:
        if stage == "fit" or stage is None:
            dataset_train = JSSS_wave(
                modes=self.modes,
                transform=self.transform,
                dir_data=self.dir_root,
                resample_sr=self._resample_sr
            )
            n_train = len(dataset_train)
            self.data_train, self.data_val = random_split(
                dataset_train, [n_train - 10, 10]
            )
        if stage == "test" or stage is None:
            self.data_test = JSSS_wave(
                modes=self.modes,
                transform=self.transform,
                dir_data=self.dir_root,
                resample_sr=self._resample_sr
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
    dm_npVCC_wave = NpVCC2016DataModule(1, download=True)

    # download & preprocessing
    dm_npVCC_wave.prepare_data()

    # runtime setup
    dm_npVCC_wave.setup(stage="fit")

    # yield dataloader
    dl = dm_npVCC_wave.train_dataloader()
    print(next(iter(dl)))
    print("datamodule/waveform.py test passed")