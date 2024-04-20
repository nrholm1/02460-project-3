import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj

def get_mutag_dataset(device:str='cpu'):
    dataset = TUDataset(root='../data/', name='MUTAG').to(device)
    return dataset

def plot_adj(adj: torch.Tensor, ax=None, name="Adjacency Matrix"):
    if ax is None:
        plt.spy(adj)
        plt.title(f"{name}, shape=[{adj.shape[0]}x{adj.shape[1]}]")
        plt.tight_layout()
    else:
        ax.spy(adj)
        ax.set_title(f"{name}, shape=[{adj.shape[0]}x{adj.shape[1]}]")
        ax.grid(False)