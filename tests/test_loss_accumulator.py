import torch

from detect_to_track.utils import DTLoss


def test_loss_accumulator_gradients():
    a_losses_lst = [
        torch.rand(1).squeeze().requires_grad_(True)
        for _ in range(5)
    ]

    b_losses_lst = [
        torch.rand(1).squeeze().requires_grad_(True)
        for _ in range(5)
    ]

    a_losses = DTLoss(*a_losses_lst)
    b_losses = DTLoss(*[2 * loss for loss in b_losses_lst])

    a_losses += b_losses

    a_losses.backward()

    for loss in a_losses_lst:
        assert loss.grad == 1 / 2

    for loss in b_losses_lst:
        assert loss.grad == 2 / 2
