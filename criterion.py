import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LmCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None, reduction='batchmean') -> None:
        super(LmCrossEntropyLoss, self).__init__()
        assert reduction in ['none', 'batchmean', 'sum', 'mean']
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = self.compute_loss(input, target)
        return self._reduce(loss)

    def compute_loss(self, input: Tensor, target: Tensor) -> Tensor:
        batch_size, _, num_embeddings = input.shape
        loss = self.criterion(
            input.view(-1, num_embeddings),
            target.view(-1)
        ).view(batch_size, -1)
        return loss

    def _reduce(self, loss: Tensor) -> Tensor:
        if self.reduction == 'batchmean':
            return loss.sum(dim=1).mean(dim=0)
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()
        return loss


class LabelSmoothedLmCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None, reduction='batchmean', label_smoothing=0.1) -> None:
        super(LabelSmoothedLmCrossEntropyLoss, self).__init__()
        assert reduction in ['none', 'batchmean', 'sum', 'mean']
        assert label_smoothing > 0

        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = self.compute_loss(input, target)
        return self._reduce(loss)

    def compute_loss(self, input: Tensor, target: Tensor) -> Tensor:
        if target.dim() == input.dim() - 1:
            target = target.unsqueeze(-1)

        lprobs = F.log_softmax(input, dim=-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)

        epsilon = self.label_smoothing
        eps_i = epsilon / lprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    def _reduce(self, loss: Tensor) -> Tensor:
        if self.reduction == 'batchmean':
            return loss.sum(dim=1).mean(dim=0)
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'mean':
            return loss.mean()
        return loss

