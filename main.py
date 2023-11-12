from argparse import Namespace
import torch
from model import get_numid_model
from utils.parser import GlobalArgsParser
from utils.device import get_device
from mnist import get_mnist_dataloader
import logging


def main(args):
    dev = get_device(args)
    train_dataloader = get_mnist_dataloader(train=True, args=args)
    test_dataloader = get_mnist_dataloader(train=False, args=args)
    model = get_numid_model(device=dev)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        train(model, train_dataloader, loss, optimizer, epoch, dev, args)
        test(model, test_dataloader, loss, epoch, dev, args)


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    args: Namespace,
) -> None:
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)

        output = model(data)
        loss_val = loss(output, target)
        loss_val.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            logging.info(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} "
                f"({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss_val.item():.6f}"
            )


def test(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss: torch.nn.Module,
    epoch: int,
    device: torch.device,
    args: Namespace,
) -> None:
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += target.size(0)
            correct += (output.argmax(dim=1) == target).sum().item()
    logging.info(f"Test Epoch: {epoch}\tAccuracy: {100. * correct / total:.2f}")


if __name__ == "__main__":
    args = GlobalArgsParser().get_args()
    logging.basicConfig(
        level=args.log_level, style="{", format="[{levelname}][{asctime}]: {message}"
    )
    main(args)
