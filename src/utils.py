import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch import Tensor

def get_mutag_dataset(device:str='cpu'):
    dataset = TUDataset(root='../data/', name='MUTAG').to(device)
    return dataset

def plot_adj(adj: Tensor, name="Adjacency Matrix"):
    # plt.spy(adj, markersize=1)
    plt.spy(adj)
    plt.title(f"{name} shape=[{adj.shape[0]}x{adj.shape[1]}]")
    plt.tight_layout()
    plt.show()
