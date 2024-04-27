import torch_geometric
import networkx as nx
import numpy as np

from src.utils import get_mutag_dataset, plot_adj
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from src.graph_statistics import GraphStatistics

def convert_geometric_to_network_x(graph: Data):
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    adj = torch_geometric.utils.to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    adj.setdiag(0)  # Remove self-loops
    adj.eliminate_zeros()  # Clean up matrix after modification

    is_symmetric = (adj != adj.T).nnz == 0

    # Create a NetworkX graph from the adjacency matrix, based on whether it is symmetric or not
    if is_symmetric:
        G = nx.Graph(adj)  # Convert to a NetworkX graph by passing the matrix directly  
    else:
        G = nx.DiGraph(adj)  

    return G

if __name__ == "__main__":
    dataset = get_mutag_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # dataloader for sampling single graphs

    # test the whole dataset one by one
    for graph in dataloader:
        G = convert_geometric_to_network_x(graph)
        # node degree
        degrees = dict(G.degree())
        # clustering coefficient
        clustering = nx.clustering(G)
        # eigenvector centrality
        centrality = nx.eigenvector_centrality(G,max_iter=10000)

        dense_adj = torch_geometric.utils.to_dense_adj(graph.edge_index).squeeze()
        graph_stats = GraphStatistics(dense_adj)

        assert np.allclose(list(degrees.values()), graph_stats.degree)
        assert np.allclose(list(clustering.values()), graph_stats.clustercoefficient)
        assert np.allclose(list(centrality.values()), graph_stats.eigenvector_centrality, atol=1e-3)


    print("All tests passed!")