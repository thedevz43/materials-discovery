import torch
from src.train import masked_mae

def test_masked_mae():
    """
    Test the masked_mae function to ensure it handles NaN values correctly.
    """
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, float('nan'), 2.0])
    loss = masked_mae(pred, target)
    assert torch.isclose(loss, torch.tensor(0.5)), f"Expected 0.5, got {loss.item()}"

def test_masked_mae_all_nan():
    """
    Test the masked_mae function when all targets are NaN.
    """
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([float('nan'), float('nan'), float('nan')])
    loss = masked_mae(pred, target)
    assert loss.item() == 0.0, f"Expected 0.0, got {loss.item()}"