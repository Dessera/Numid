import torch
import logging


class NumidModel(torch.nn.Module):
    model_stack: torch.nn.Sequential

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # use 5 layers conv2d
        self.model_stack = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model_stack(x)


def get_numid_model(device: torch.device) -> NumidModel:
    """获取NumidModel

    Args:
        device (torch.device): 当前使用设备

    Returns:
        NumidModel: NumidModel实例
    """
    logging.info("Loading NumidModel")
    model = NumidModel()
    model.to(device)
    return model
