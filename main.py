import math
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from matplotlib import pyplot as plt

batches_per_epoch = 100
batch_size = 65536
epochs = 30

lr = 1e-2

num_plots = 4
epochs_per_plot = epochs // (num_plots-1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training using {device}...")


class Model(torch.nn.Module):
    def __init__(self,
                 in_features=1,
                 hidden_dim=5,
                 out_features=1,
                 n_hidden_layers=0,
                 ):
        super().__init__()

        layers = nn.Sequential()

        layers.add_module('input_layer', nn.Linear(in_features, hidden_dim))
        layers.add_module('input_activation', nn.Sigmoid())

        for i in range(n_hidden_layers):
            layers.add_module(f'hidden_layer{i}', nn.Linear(hidden_dim, hidden_dim))
            layers.add_module(f'hidden_activation{i}', nn.Sigmoid())

        layers.add_module('output_layer', nn.Linear(hidden_dim, out_features))

        self.mlp = layers

    def forward(self, x):
        return self.mlp(x)


# The function we want to learn
f = torch.sin


# f = lambda x:  torch.sin(x) if x > 1.0 else 1 / (1 + torch.exp(-x))

# Generate a batch of x, f(x) pairs for training/testing
def gen_batch(batch_size):
    x = 10 * (2. * torch.rand(batch_size, device=device) - 1)
    y = f(x)
    yield x, y


# initialize the model and display its architecture
mlp = Model().to(device)
print(mlp)
# initialize optimizer and loss function
# opt = torch.optim.SGD(mlp.parameters(), lr=LR)
opt = torch.optim.Adam(mlp.parameters(), lr=lr)
lossFunc = nn.MSELoss()

plot_num = 0

scheduler = StepLR(opt, step_size=1, gamma=0.99)

plt.figure(figsize=(num_plots+1, 2))

losses = []
for epoch in tqdm(range(0, epochs)):

    trainLoss = 0

    mlp.train()

    for _ in range(batches_per_epoch):
        x, y = next(gen_batch(batch_size))
        y_hat = mlp(torch.unsqueeze(x, -1))
        loss = lossFunc(y_hat, torch.unsqueeze(y, -1))
        # zero the gradients accumulated from the previous steps,
        # perform backpropagation, and update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    if (epoch % epochs_per_plot) == 0 or epoch == epochs-1:
        plot_num = plot_num + 1
        plt.subplot(2, num_plots, plot_num)
        mlp.eval()
        plt.title(f"epoch: {epoch + 1}", fontsize=10)
        x = torch.linspace(-math.pi * 2, math.pi * 2, 1000, device=device)
        y = f(x)
        y_hat = torch.squeeze(mlp(torch.unsqueeze(x, -1)))

        x = x.cpu().numpy()
        y = y.cpu().numpy()
        y_hat = y_hat.cpu().detach().numpy()
        plt.plot(x, y, color='green')
        plt.plot(x, y_hat, color='red')

    scheduler.step()
plt.subplot(2, 1, 2)
plt.title(f"Loss: {losses[-1]: .4f}", fontsize=10)
plt.plot(range(len(losses)), losses)
print(losses[-1])
plt.show()
