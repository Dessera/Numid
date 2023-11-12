from torch import device as torch_device
from torch import cuda as torch_cuda
import logging
from argparse import Namespace


def get_device(args: Namespace) -> torch_device:
    logging.info(f"Using device {args.device}")
    if torch_cuda.is_available():
        return torch_device(args.device)
    elif args.device == "cuda":
        logging.warning("CUDA is not available, using CPU")
        return torch_device("cpu")
    else:
        return torch_device("cpu")
