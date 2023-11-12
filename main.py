from argparse import Namespace
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from model import get_numid_model
from utils.parser import GlobalArgsParser
from utils.device import get_device
from mnist import get_mnist_dataloader
import logging
import os


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

    accuracies = []

    for epoch in range(args.epochs):
        train(model, train_dataloader, loss, optimizer, epoch, dev, args)
        acc = test(model, test_dataloader, loss, epoch, dev, args)
        accuracies.append(acc)
    logging.info(f"Final accuracy: {accuracies[-1]:.2f}")

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    figure_save_path = os.path.join(args.figure, f"{current_time}.png")
    plot_epoch_accuracy(accuracies, save_path=figure_save_path)

    model_save_path = os.path.join(args.model, f"{current_time}.pt")
    logging.info(f"Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)


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
) -> float:
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += target.size(0)
            correct += (output.argmax(dim=1) == target).sum().item()

    accuracy = 100.0 * correct / total
    logging.info(f"Test Epoch: {epoch}\tAccuracy: {accuracy:.2f}")
    return accuracy


def plot_epoch_accuracy(accuracies: list[float], save_path: str | None = None) -> None:
    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Epoch Accuracy")
    # plt.show()
    if save_path:
        logging.info(f"Saving figure to {save_path}")
        plt.savefig(save_path)
    else:
        logging.info("Plot epoch-accuracy figure")
        plt.show()


if __name__ == "__main__":
    args = GlobalArgsParser().get_args()
    logging.basicConfig(
        level=args.log_level, style="{", format="[{levelname}][{asctime}]: {message}"
    )
    main(args)
