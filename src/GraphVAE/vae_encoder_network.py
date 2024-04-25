import torch
import torch.nn as nn

class GNNEncoderNetwork(nn.Module):
    def __init__(self, node_feature_dim: int, embedding_dim: int, n_message_passing_rounds: int, M: int) -> None:
        """
        A message passing GNN used to parameterize the encoder of the VAE

        """
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.embedding_dim = embedding_dim
        self.n_message_passing_rounds = n_message_passing_rounds
        self.M = M
        self.embedding_network = nn.Sequential(nn.Linear(self.node_feature_dim, self.embedding_dim),
                                               nn.ReLU())
        
        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, self.embedding_dim),
                torch.nn.ReLU()
            ) for _ in range(self.n_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, self.embedding_dim),
                torch.nn.ReLU()
            ) for _ in range(self.n_message_passing_rounds)])

        # State output network
        self.output_net = torch.nn.Linear(self.embedding_dim, self.M)
    
    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1
        num_nodes = batch.shape[0]

        # Initialize node state from node features
        state = self.embedding_network(x)

        # Loop over message passing rounds
        for r in range(self.n_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.embedding_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])
            
            # Update states
            state = state + self.update_net[r](aggregated) # skip connection

        # Aggretate: Sum node features
        graph_state = x.new_zeros((num_graphs, self.embedding_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)
        out = self.output_net(graph_state)
        return out

class GRUGNNEncoderNetwork(torch.nn.Module):
    """Simple graph neural network for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        state_dim : Dimension of the node states
        num_message_passing_rounds : Number of message passing rounds
    """

    def __init__(self, node_feature_dim: int, embedding_dim: int, num_message_passing_rounds: int, M: int):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.embedding_dim = embedding_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.M = M

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.embedding_dim),
            torch.nn.ReLU()
            )
        
        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, self.embedding_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])
    
        # GRU update network
        self.W_mr = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_mz = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_mh = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.W_hr = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_hz = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_hh = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

        # State output network
        self.output_net = torch.nn.Linear(self.embedding_dim, self.M)
    
    def reset(self, message, h):
        return self.sigmoid(self.W_mr(message) + self.W_hr(h))
        
    def update(self, message, h):
        return self.sigmoid(self.W_mz(message) + self.W_hz(h))

    def candidate(self, message, h, r):
        return self.tanh(self.W_mh(message) + self.W_hh(r * h))
    
    def GRU_update(self, message, h_u):
        r = self.reset(message, h_u)
        z = self.update(message, h_u)
        h = self.candidate(message, h_u, r)
        return z * h + (1 - z) * h_u

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

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
            aggregated = x.new_zeros((num_nodes, self.embedding_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            # state = state + self.update_net[r](aggregated) # skip connection
            state = self.GRU_update(h_u=state, message=message)

        # Aggretate: Sum node features
        graph_state = x.new_zeros((num_graphs, self.embedding_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)

        # Output
        out = self.output_net(graph_state)
        return out
