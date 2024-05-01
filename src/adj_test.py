#%%
import torch
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
# %%
plot_adj = lambda adj: plt.spy(adj)
plot_edge_index = lambda edge_index: plot_adj(to_dense_adj(edge_index).squeeze())
def plot_adj_color(matrix): plt.imshow(matrix); plt.colorbar(); plt.show()
plot_edge_index_color = lambda edge_index: plot_adj_color(to_dense_adj(edge_index).squeeze())
# %%
num_nodes_full = 10
triu_idx_full = torch.triu_indices(num_nodes_full,num_nodes_full,1)

num_nodes = 6
triu_idx = torch.triu_indices(num_nodes,num_nodes,1)

triu_idx[1] += (num_nodes_full - num_nodes)

dense_adj = torch.zeros(num_nodes_full, num_nodes_full)

def compute_diag_idx(diag_size: int, full_size: int = None):
    full_size = full_size or diag_size # if full_size none, compute plain diag
    idx0 = torch.arange(diag_size).unsqueeze(0)
    idx1 = idx0 + (full_size - diag_size)
    return torch.vstack([idx0, idx1])

# perform this with pure torch operations?
batch_diag_idx = torch.cat([
    compute_diag_idx(cur_n, num_nodes)
    for cur_n in range(1, num_nodes)
], dim=1)


dense_adj[batch_diag_idx[0], batch_diag_idx[1]] = torch.arange(1, num_nodes*(num_nodes-1) / 2 + 1)

plot_adj_color(dense_adj)

