import torch
import torch.optim as optim
from tqdm import tqdm
import pdb

class VAETrainer:
    def __init__(self, model, optimizer, data_loader, epochs, device) -> None:
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.epochs = epochs
        self.device = device
        self.model.to(self.device)
    
    def train(self):
        """
        Train a VAE model.

        Parameters:
        model: [VAE]
        The VAE model to train.
        optimizer: [torch.optim.Optimizer]
            The optimizer to use for training.
        data_loader: [torch.utils.data.DataLoader]
                The data loader to use for training.
        epochs: [int]
            Number of epochs to train for.
        device: [torch.device]
            The device to use for training.
        """

        self.model.train()
        num_steps = len(self.data_loader) * self.epochs
        epoch = 0
        losses = []
        
        with tqdm(range(num_steps)) as pbar:
            for step in pbar:
                data = next(iter(self.data_loader))
                data = data.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model(data.x, data.edge_index, batch=data.batch)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

                # Report
                if step % 5 ==0 :
                    loss = loss.detach().cpu()
                    pbar.set_description(f"epoch={epoch}, step={step}, loss={torch.mean(torch.tensor(losses)):.1f}")

                if (step+1) % len(self.data_loader) == 0:
                    epoch += 1

        