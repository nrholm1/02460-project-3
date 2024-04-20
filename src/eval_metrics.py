#%%
import torch
from torch_geometric.utils import to_dense_adj
from networkx import weisfeiler_lehman_graph_hash, from_numpy_array as nx_from_numpy_array

def adj_tensor_to_nx_graph(adj: torch.Tensor):
    adj = adj.detach().numpy() # detach (if for any reason there is a grad)
    G = nx_from_numpy_array(adj)
    return G


def ensure_adj(g: torch.Tensor) -> torch.Tensor:
    if (g.shape[0] != g.shape[1]): g = to_dense_adj(g).squeeze()
    return g


def isomorphic(g1: torch.Tensor, g2: torch.Tensor) -> bool:
    g1 = ensure_adj(g1)
    g2 = ensure_adj(g2)

    G1 = adj_tensor_to_nx_graph(g1)
    G2 = adj_tensor_to_nx_graph(g2)

    # compute graph hashes, which are identical between two graphs iff they are isomorphic
    # ? Note: in theory you could have hash collisions, but probability is negligible
    gh1 = weisfeiler_lehman_graph_hash(G1)
    gh2 = weisfeiler_lehman_graph_hash(G2)
    return gh1 == gh2


def compute_graph_hashes(graphs, return_list=False) -> set[str] | list[str]:
    """
    Compute either set or list of hashes.
    """
    hashes = []
    for graph in graphs:
        G = ensure_adj(graph)
        G = adj_tensor_to_nx_graph(G)
        gh = weisfeiler_lehman_graph_hash(G)
        hashes.append(gh)

    return hashes if return_list else set(hashes)


def eval_novelty(gen_graph_hashes: list[str], true_graph_hashes: set[str]) -> float:
    """
    Count how many of the generated graph hashes are not in the set of ground truth graph hashes.
    """
    return torch.mean(torch.tensor([hash not in true_graph_hashes for hash in gen_graph_hashes]).float()).item()


def eval_unique(gen_graph_hashes: list[str]) -> float:
    # TODO: correctly interpreted?
    """
    Count percentage of sampled graphs that are unique, i.e. only exist once in the set.
        Ex.: [a,a,b,c] -> 0.5, since only b,c are unique.
    """
    id_map = {hash: idx for idx, hash in enumerate(set(gen_graph_hashes))}
    mapped_hash_list = [id_map[h] for h in gen_graph_hashes]
    hash_counts = torch.tensor(mapped_hash_list).unique(return_counts=True)[1]
    num_appear_once = torch.sum(hash_counts == 1)
    return (num_appear_once / len(gen_graph_hashes)).item()



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.utils import plot_adj

    adj1 = torch.tensor([
        [1,0],
        [1,1],
    ])
    adj2 = torch.tensor([
        [1,1],
        [0,1],
    ])

    are_isomorphic = isomorphic(adj1,adj2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_adj(adj1, axs[0], name="Adj. 1")
    plot_adj(adj2, axs[1], name="Adj. 2")
    print(f"Test graphs - They are {'NON-' if not are_isomorphic else ''}isomorphic")
    plt.show()