"""Exact DQN architecture used in the final five-seed experiment."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAVING_SHARES = np.arange(0.0, 1.0, 0.05, dtype=np.float64)
A_MAX = 4.5

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(SAVING_SHARES))

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def load_dqn(checkpoint_path, device="cpu"):
    model = QNetwork().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def dqn_saving_share(model, assets, incomes):
    """Return saving shares for one or many states: assets and incomes are broadcastable."""
    assets, incomes = np.broadcast_arrays(np.asarray(assets), np.asarray(incomes))
    obs = np.column_stack([assets.ravel() / A_MAX, incomes.ravel()]).astype(np.float32)
    with torch.no_grad():
        actions = model(torch.from_numpy(obs)).argmax(dim=1).numpy()
    return SAVING_SHARES[actions].reshape(assets.shape)
