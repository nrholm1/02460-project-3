import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, to_dense_batch
from src.utils import get_mutag_dataset
from torch_geometric.loader import DataLoader
import pdb

class GAN_MPNN(nn.Module):
    """Simple graph neural network (message passing variant). Code from exercises week 10.

    
    Args:
        node_feature_dim (int): Dimension of the node features
        state_dim (int): Dimension of the node states
        num_message_passing_rounds (int): Number of message passing rounds
    """

    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU()
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # State output network
        # self.output_net = torch.nn.Linear(self.state_dim, 1)

    def get_input_grad(self):
        return self.last_init_state.grad # ! how to accumulate gradients from message/update nets? I.e. quite complex
        # raise NotImplementedError("Not implemented for message passing net yet!")


    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Args:
            x : torch.tensor (num_nodes x num_features)
                Node features.
            edge_index : torch.tensor (2 x num_edges)
                Edges (to-node, from-node) in all graphs.
            batch : torch.tensor (num_nodes)
                Index of which graph each node belongs to.
        
        Returns:
            torch tensor: Neural network output for each graph.
        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1
        num_nodes = batch.shape[0]

        # Initialize node state from node features
        # state = self.input_net(x)
        state = x.new_zeros([num_nodes, self.state_dim]) # Uncomment to disable the use of node features
        state.requires_grad = True
        self.last_init_state = state # used for grad

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            state = state + self.update_net[r](aggregated)

        # Aggretate: Sum node features
        graph_state = x.new_zeros((num_graphs, self.state_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)

        # Output
        # out = self.output_net(graph_state).flatten()
        return graph_state


class MessagePassingNN(nn.Module):
    """Simple graph neural network for graph classification. Code from exercises week 10

    
    Args:
        node_feature_dim (int): Dimension of the node features
        state_dim (int): Dimension of the node states
        num_message_passing_rounds (int): Number of message passing rounds
    """

    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU()
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # State output network
        self.output_net = torch.nn.Linear(self.state_dim, 1)

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Args:
            x : torch.tensor (num_nodes x num_features)
                Node features.
            edge_index : torch.tensor (2 x num_edges)
                Edges (to-node, from-node) in all graphs.
            batch : torch.tensor (num_nodes)
                Index of which graph each node belongs to.
        
        Returns:
            torch tensor: Neural network output for each graph.
        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1
        num_nodes = batch.shape[0]

        # Initialize node state from node features
        state = self.input_net(x)
        # state = x.new_zeros([num_nodes, self.state_dim]) # Uncomment to disable the use of node features

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            state = state + self.update_net[r](aggregated)

        # Aggretate: Sum node features
        graph_state = x.new_zeros((num_graphs, self.state_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)

        # Output
        out = self.output_net(graph_state).flatten()
        return out


class GraphConvNN(nn.Module):
    """Simple graph convolution for graph classification. Code from exercises week 11

    Args:
        node_feature_dim (int): Dimension of the node features
        filter_length (int): Number of filter taps
    """

    def __init__(self, node_feature_dim, filter_length):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length

        # Define graph filter
        self.h = torch.nn.Parameter(1e-5*torch.randn(filter_length))
        self.h.data[0] = 1.

        # State output network
        self.output_net = torch.nn.Linear(self.node_feature_dim, 1)

        self.cached = False

    def get_input_grad(self):
        raise NotImplementedError("Not implemented for GCN net yet!")
        # return self.h.grad

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Args:
            x (torch.tensor): Node features.
            edge_index (torch.tensor): Edges (to-node, from-node) in all graphs.
            batch (torch.tensor): Index of which graph each node belongs to.
            

        Returns:
            torch tensor: Neural network output for each graph.
        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1

        # Compute adjacency matrices and node features per graph
        A = to_dense_adj(edge_index, batch)
        X, idx = to_dense_batch(x, batch)
 

        # ---------------------------------------------------------------------------------------------------------

        # Implementation in vertex domain
        # node_state = torch.zeros_like(X)
        # for k in range(self.filter_length):
        #     node_state += self.h[k] * torch.linalg.matrix_power(A, k) @ X

        # Implementation in spectral domain
        L, U = torch.linalg.eigh(A)        
        exponentiated_L = L.unsqueeze(2).pow(torch.arange(self.filter_length, device=L.device))
        diagonal_filter = (self.h[None,None] * exponentiated_L).sum(2, keepdim=True)
        node_state = U @ (diagonal_filter * (U.transpose(1, 2) @ X))

        # ---------------------------------------------------------------------------------------------------------

        # Aggregate the node states
        graph_state = node_state.sum(1)

        # Output
        out = self.output_net(graph_state).flatten()
        return out





if __name__ == "__main__":

    data = get_mutag_dataset()
    dataloader = DataLoader(data, batch_size=10, shuffle=True)



    # Define a simple graph neural network
    node_feature_dim = 7
    state_dim = 16
    num_message_passing_rounds = 5
    model = MessagePassingNN(node_feature_dim, state_dim, num_message_passing_rounds)
    pdb.set_trace()