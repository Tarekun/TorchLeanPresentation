import torch
import torch.nn as nn
import torch.optim as optim
from data import generate_batch
from mlp import Mlp


def train(model: Mlp, train_steps: int, batch_size: int, device: torch.device = torch.device("cpu")):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for step in range(train_steps):
        x, y = generate_batch(batch_size, device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"step {step + 1}/{train_steps}  loss={loss.item():.4f}")

def evaluate(model: Mlp, test_size: int, device: torch.device = torch.device("cpu")):
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        x, y = generate_batch(test_size, device)
        pred = model(x)
        loss = loss_fn(pred, y)
    print(f"eval  mse={loss.item():.4f}")
