import pathlib as path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataloader import IDAODataset, img_loader, InferenceDataset


class IDAODataModule(pl.LightningDataModule):
    def __init__(self, data_dir: path.Path, batch_size: int, cfg):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.cfg = cfg

    def prepare_data(self):
        # called only on 1 GPU
        self.dataset = IDAODataset(
            root=self.data_dir.joinpath("train"),
            loader=img_loader,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.CenterCrop(64)]
            ),
            # TODO(kazeevn) use idiomatic torch
            target_transform=transforms.Compose(
                [
                    lambda num: (
                        torch.tensor([0, 1]) if num == 0 else torch.tensor([1, 0])
                    )
                ]
            ),
            extensions=self.cfg["DATA"]["Extension"],
        )
        
        self.val = IDAODataset(
            root=self.data_dir.joinpath("val"),
            loader=img_loader,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.CenterCrop(64)]
            ),
            # TODO(kazeevn) use idiomatic torch
            target_transform=transforms.Compose(
                [
                    lambda num: (
                        torch.tensor([0, 1]) if num == 0 else torch.tensor([1, 0])
                    )
                ]
            ),
            extensions=self.cfg["DATA"]["Extension"],
        )

        self.test = InferenceDataset(
                    main_dir=self.data_dir.joinpath("test"),
                    loader=img_loader,
                    transform=transforms.Compose(
                        [transforms.ToTensor(), transforms.CenterCrop(64)]
                    ),
                )


    def setup(self, stage=None):
        # called on every GPU
        self.train = self.dataset

    def train_dataloader(self):
        train = DataLoader(self.train, self.batch_size, shuffle=True, num_workers=24)
        #print(print(len(train.dataset)))
        return train

    def val_dataloader(self):
        return DataLoader(self.val, 1, num_workers=24, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(
            self.test,
            self.batch_size,
            num_workers=24,
            shuffle=False
            )

