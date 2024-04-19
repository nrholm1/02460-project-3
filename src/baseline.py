#%%

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from src.utils import get_mutag_dataset, plot_adj

class Baseline(torch.nn.Module):
    """
    Implements the Erdös-Rényi graph generation model.
    """
    def __init__(self, dataset):
        super().__init__()
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        self.N_dist, self.r_dist = self.compute_graph_statistics(dataloader)

    def compute_graph_statistics(self, dataloader):
        # ! load all graphs in dataset for computing graph stats
        graphs = next(iter(dataloader))
        num_graphs = len(graphs)

        # compute empirical distribution of N, i.e. #nodes
        graph_idxs = torch.arange(num_graphs).unsqueeze(1)
        mask = graphs.batch == graph_idxs
        nodes_per_graph = (mask).sum(1) # how many nodes are in each graph
        N_dist = torch.vstack(torch.unique(nodes_per_graph, return_counts=True))
        N_dist = N_dist.float()
        N_dist[1] /= num_graphs # convert counts to probabilities

        # find uniform edge probabilities for each graph size (N)
        src_counts = (torch.unique(graphs.edge_index, return_counts=True)[1].repeat(num_graphs,1) * mask.float()).sum(1) # no. of edges with its corresponding node idx as source
        edge_probs_unif = src_counts / nodes_per_graph**2 # ! assuming self-loops are allowed, (max possible #edges) = N^2
        probs_given_N = (edge_probs_unif * (nodes_per_graph == N_dist[0].unsqueeze(1)).float()).mean(1) # compute mean of uniform edge prob for each possible N
        r_dist = torch.vstack([N_dist[0], probs_given_N])

        return N_dist, r_dist
    
    def sample_graph(self):
        # sample N according to computed empirical distribution
        N_idx = torch.multinomial(self.N_dist[1], num_samples=1)
        N = self.N_dist[0, N_idx].int().item()

        # reuse index sampled for N, since the distributions are indexed identically by design
        r = self.r_dist[1, N_idx]

        # generate adjacency matrix
        adj = torch.rand((N,N))
        link_mask = adj <= r # select indices that pass the generation criteria
        adj[link_mask] = 1
        adj[~link_mask] = 0

        # ! return edge index instead?
        return adj.int() 

    
    def forward(self):
        """
        Generate a graph according to the genrative process:
            1. sample according to empirical distribution computed for initialization dataset.
            2. generate links according to probability r, computed also for initialization dataset. 
        """
        return self.sample_graph()

# %%
dataset = get_mutag_dataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
sample_adj = lambda: to_dense_adj(next(iter(dataloader)).edge_index).squeeze()

model = Baseline(dataset)

# %%

# %%
plot_adj(sample_adj(), name="Sampled graph")
plot_adj(model(), name="Generated graph")


