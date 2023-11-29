import torch


class ZeroTargetHuberLoss(torch.nn.Module):
    def __init__(self, delta=1.):
        super().__init__()
        self.delta = delta

    def forward(self, prediction, target):
        mask = (target == 0).float()
        error = mask * prediction
        abs_error = error.abs()
        quadratic = torch.min(abs_error, torch.full_like(abs_error, self.delta))
        linear = (abs_error - quadratic)
        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean(dim=-1, keepdim=True)
