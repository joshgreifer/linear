import math
import torch.nn as nn
import torch
import torchsummary
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from matplotlib import pyplot as plt

input_size = 16

batches_per_epoch = 1000
batch_size = 16
epochs = 100

lr = 1e-3

num_plots = 5
epochs_per_plot = epochs // (num_plots-1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training using {device}...")


class Model(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=16,
                 n_hidden_layers=4,
                 ):
        super().__init__()

        layers = nn.Sequential()

        layers.add_module('input_layer', nn.Linear(input_size, hidden_size))
        layers.add_module('input_activation', nn.GELU())

        for i in range(n_hidden_layers):
            layers.add_module(f'hidden_layer{i}', nn.Conv1d(hidden_size, 2 * hidden_size, (8,)))
            layers.add_module(f'hidden_activation{i}', nn.GELU())

        layers.add_module('output_layer', nn.Linear(hidden_size, input_size))

        self.mlp = layers

    def forward(self, x):
        for i in range(self.mlp.__len__()):
            x = self.mlp[i](x)
            print(x.shape)

        return x
        # return self.mlp(x)


# The function we want to learn
f = torch.fft.fft


# f = lambda x:  torch.sin(x) if x > 1.0 else 1 / (1 + torch.exp(-x))

# Generate a batch of x, f(x) pairs for training/testing
def gen_batch(batch_size, input_size):
    x = 2. * torch.rand((batch_size, input_size), device=device) - 1
    y = f(x)
    yield x, y


# initialize the model and display its architecture
mlp = Model(input_size=input_size).to(device)
print(mlp)
# initialize optimizer and loss function
# opt = torch.optim.SGD(mlp.parameters(), lr=LR)
opt = torch.optim.Adam(mlp.parameters(), lr=lr)


def complex_mse_loss(output, target):
    return (0.5*(output - target)**2).mean(dtype=torch.complex64)


lossFunc = complex_mse_loss


testTemplate = "epoch: {} train loss: {:.6f} LR: {:.4f}"
plot_num = 0

scheduler = StepLR(opt, step_size=1, gamma=0.99)

plt.figure(figsize=(1, num_plots+1))
pbar = tqdm(range(0, epochs))
for epoch in pbar:

    trainLoss = 0
    samples = 0

    mlp.train()

    for _ in range(batches_per_epoch):
        x, y = next(gen_batch(batch_size, input_size))
        y_hat = mlp(x)
        loss = lossFunc(y_hat, y)
        # zero the gradients accumulated from the previous steps,
        # perform backpropagation, and update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        trainLoss += torch.abs(loss).item() * y.size(0)
        samples += y.size(0)
    stats_info = f"epoch: {epoch + 1} train loss: {trainLoss / samples:.4f} LR: {scheduler.get_last_lr()[0]:.4f}"
    pbar.set_description(stats_info)
    
    scheduler.step()

