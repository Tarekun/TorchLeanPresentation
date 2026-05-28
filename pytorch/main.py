import torch
from mlp import Mlp
from training import train, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

model = Mlp().to(device)
train(model, train_steps=100, batch_size=32, device=device)
evaluate(model, test_size=200, device=device)
