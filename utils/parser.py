import argparse


class GlobalArgsParser:
    parser: argparse.ArgumentParser

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="Pytorch implementation of Handwritten Numeral Recognition"
        )
        self.parser.add_argument(
            "--data",
            type=str,
            default="data",
            help="path to dataset (default: data)",
        )
        self.parser.add_argument(
            "--model",
            type=str,
            default="model",
            help="path to save model (default: model)",
        )
        self.parser.add_argument(
            "--figure",
            type=str,
            default="figure",
            help="path to save figure (default: figure)",
        )
        self.parser.add_argument(
            "--log_level",
            type=str,
            default="INFO",
            help="logging level (default: INFO)",
        )
        self.parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="use cpu or cuda (default: cuda)",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="number of workers (default: 4)",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="batch size (default: 64)",
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="number of epochs (default: 10)",
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="learning rate (default: 0.01)",
        )
        self.parser.add_argument(
            "--momentum",
            type=float,
            default=0.9,
            help="SGD momentum (default: 0.9)",
        )
        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=5e-4,
            help="weight decay (default: 5e-4)",
        )
        self.parser.add_argument(
            "--print_freq",
            type=int,
            default=10,
            help="print frequency (default: 10)",
        )

    def get_args(self) -> argparse.Namespace:
        return self.parser.parse_args()
