#%%
from matplotlib import pyplot as plt
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

        # find uniform edge probabilities for each graph size (N)
        src_counts = (torch.unique(graphs.edge_index[0], return_counts=True)[1].repeat(num_graphs,1) * mask.float()).sum(1) # no. of edges with its corresponding node idx as source
        edge_probs_unif = src_counts / (nodes_per_graph*(nodes_per_graph-1)) # ! assuming self-loops are not allowed, (max possible #edges) = N*(N-1)
        probs_given_N = (edge_probs_unif * (nodes_per_graph == N_dist[0].unsqueeze(1)).float()).sum(1) / N_dist[1] # compute mean of uniform edge prob for each possible N
        r_dist = torch.vstack([N_dist[0], probs_given_N])

        N_dist[1] /= num_graphs # finally, convert counts to probabilities (we needed to use it to compute probs_given_N)

        return N_dist, r_dist
    

    def sample_edge_index(self, undirected: bool):
        """
        Samples a single graph (unbatcheed) in the sparse representation, i.e. edge_index.
        """
        # sample N according to computed empirical distribution
        N_idx = torch.multinomial(self.N_dist[1], num_samples=1)
        N = self.N_dist[0, N_idx].int().item()

        # reuse index sampled for N, since the distributions are indexed identically by design
        r = self.r_dist[1, N_idx]

        if undirected:
            # r *= 2 # ! double probability for undirected, since we only assign to half ??
            upper_tri = torch.triu_indices(N,N,1)
            link_mask = torch.rand(upper_tri.shape[1]) <= r # select indices in upper triangular part that pass the generation criteria
            upper_tri = upper_tri[:, link_mask]
            edge_index = torch.hstack([
                upper_tri, torch.vstack([upper_tri[1],upper_tri[0]])
            ])
        else: # not symmetric
            upper_tri = torch.triu_indices(N,N,1)
            lower_tri = torch.tril_indices(N,N,-1)
            upper_link_mask = torch.rand(upper_tri.shape[1]) <= r # select indices in upper triangular part that pass the generation criteria
            lower_link_mask = torch.rand(lower_tri.shape[1]) <= r # select indices in lower triangular part that pass the generation criteria
            upper_tri = upper_tri[:, upper_link_mask]
            lower_tri = lower_tri[:, lower_link_mask]
            edge_index = torch.hstack([upper_tri,lower_tri])

        return edge_index


    def forward(self, batch_size=1, undirected=True, return_adj=False):
        """
        Generate a graph according to the genrative process:
            1. sample according to empirical distribution computed for initialization dataset.
            2. generate links according to probability r, computed also for initialization dataset. 
        """
        assert batch_size == 1, "Batched sampling not implemented!"
        edge_index = self.sample_edge_index(undirected=undirected)
        if return_adj: return to_dense_adj(edge_index)
        else: return edge_index


if __name__ == '__main__':
    dataset = get_mutag_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # dataloader for sampling single graphs
    sample_adj = lambda: to_dense_adj(next(iter(dataloader)).edge_index).squeeze() # sample and convert to dense adjacency matrix
    sample_model = lambda _model: to_dense_adj(_model()).squeeze()
    sample_model_directed = lambda _model: to_dense_adj(_model(undirected=False)).squeeze()

    baseline_model = Baseline(dataset)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    plot_adj(sample_adj(), axs[0], name="Sampled Graph")
    plot_adj(sample_model(baseline_model), axs[1], name="Generated Graph")
    # plot_adj(sample_model_directed(baseline_model), axs[1], name="Generated Graph")

    plt.show()


