import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from argparse import Namespace

import os
import logging


def get_mnist_dataloader(
    train: bool, args: Namespace
) -> DataLoader:
    mode = "train" if train else "test"
    logging.info(f"Loading {mode} FashionMNIST dataset")
    return DataLoader(
        datasets.FashionMNIST(
            args.data,
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
