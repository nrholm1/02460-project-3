
import torch
import torch.nn as nn
import torch.distributions as td
from torch_geometric.utils import to_dense_adj, to_dense_batch
import pdb
#from torch_geometric.nn import global_mean_pool

"""
Goal is to train a probabilistic decoder model p_theta(A|Z), from which we can sample realistic graphs
by conditioning on a latent variable z. The model is trained on a dataset of graphs, where
"""
# take a max graph size
# add padding on decoder'

def filter_edges(edge_index, node_mask):
    # This function filters edges to those that connect nodes within the node_mask
    mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    return edge_index[:, mask]

class GaussianPrior(nn.Module):
    def __init__(self, latent_dim):
        """
        Define a Gaussian prior distribution with zero latent_dimean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.mean = nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoder(nn.Module): # from week 1
    def __init__(self, mu_encoder_net, std_encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super().__init__()
        self.mu_encoder_net = mu_encoder_net
        self.std_encoder_net = std_encoder_net

    def forward(self, x, edge_index, batch):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        #mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        mu = self.mu_encoder_net(x, edge_index, batch)
        std = self.std_encoder_net(x, edge_index, batch) #should give the log std 

        return td.Independent(td.Normal(loc=mu, scale=torch.exp(std)+1e-6), 1)
        
class BernoulliDecoder(nn.Module): # from week 1
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super().__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        # thresholding?
        #return td.Independent(td.Bernoulli(logits=logits), 2)

        # Return probabilities
        #probs = torch.sigmoid(logits)
        return logits

class GNNEncoder(torch.nn.Module): # modification from week 10
    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, GRU=False, latent_dim=8):
        """Modification of Simple graph neural network

        Keyword Arguments
        -----------------
            node_feature_dim : Dimension of the node features (which is 7 in this case) which atom (carbon, hydrogen)
            state_dim : Dimension of the node states #(which is 16 in this case)
            num_message_passing_rounds : Number of message passing rounds to perform (such as 4 in this case)
        """
        super().__init__()
        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim 
        self.state_dim = state_dim 
        self.num_message_passing_rounds = num_message_passing_rounds
        self.GRU = GRU
        self.latent_dim = latent_dim

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU())

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(2 * self.state_dim, self.state_dim), # Combining features of two connected nodes
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])
        
        #Update network
        if self.GRU: #use flag to switch between GRU and linear
            self.update_net = torch.nn.ModuleList([
                torch.nn.GRUCell(self.state_dim, self.state_dim)
                for _ in range(num_message_passing_rounds)])
        else: #use simple MLP
            self.update_net = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(self.state_dim, self.state_dim),
                    torch.nn.ReLU()) 
                for _ in range(num_message_passing_rounds)])

        # State output network
        self.output_net = torch.nn.Linear(self.state_dim, self.latent_dim) # 2*latent_dim
        # (num_graphs, num nodes, latent dim)
        # output a single value for each graph
    
    # def to_device(self, device):
    #     self.to(device)

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

        Returns (output changed)
        -------
        mu_z or log sigma_z: of size (|V|, d) 
            |V| is the number of nodes in input graph
            d is the dimension of the latent space
        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1
        num_nodes = batch.shape[0]
        print("num nodes ", num_nodes)
        print("num graphs ", num_graphs)
        print("x shape ", x.shape)
        print("edge_index shape ", edge_index.shape)
        print("batch shape ", batch.shape)

        # Initialize node state from node features
        state = self.input_net(x)

        for r in range(self.num_message_passing_rounds):
            messages = torch.zeros_like(state)
            # Accumulate messages for each node
            for src, dest in edge_index.t():
                msg_input = torch.cat([state[src], state[dest]], dim=-1)
                messages[dest] += self.message_net[r](msg_input)

            if self.GRU:
                for idx in range(num_nodes):
                    state[idx] = self.update_net[r](messages[idx], state[idx])
            else:
                state += messages  # update states

        # # Loop over message passing rounds
        # for r in range(self.num_message_passing_rounds):
        #     # Compute outgoing messages
        #     message = self.message_net[r](state)

        #     # Aggregate: Sum messages
        #     aggregated = x.new_zeros((num_nodes, self.state_dim)) 
        #     aggregated = torch.index_add(state, 0, edge_index[1], message[edge_index[0]])
        #     # torch.index_add takes a dim (0 meaning rows), and index (the rows to add to)
        #     #   and adds the message (values) to the state at the index

        #     # Update states
        #     state = state + self.update_net[r](aggregated) #uses residual connections (skip connection)

        """# Message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Create an empty tensor to collect messages
            messages = torch.zeros_like(state)
            for i, (src, dest) in enumerate(zip(*edge_index)):
                # Calculate messages for each edge and accumulate them
                messages[dest] += self.message_net[r](torch.cat([state[src], state[dest]], dim=-1))

            # Update node states using messages
            if self.GRU:
                # GRU requires a different update mechanism
                for idx in range(state.shape[0]):
                    state[idx] = self.update_net[r](messages[idx], state[idx])
            else:
                # Simple MLP update
                state += self.update_net[r](messages)"""

        # Output
        print("state shape ", state.shape)
        latent_embeddings = self.output_net(state)#.flatten()
        print("encoder out shape ", latent_embeddings.shape)
        return latent_embeddings # (num_graphs, num nodes, latent dim)
    
    # the network architecture is as follows:
    # 1. Input network: A single linear layer followed by a ReLU activation function 
    #       to map the node features 7 to the node inital states 16
    # 2. Message network: A list of num_message_passing_rounds linear layers followed by ReLU activation functions
    #       from the node states to compute the messages.
    # 3. Update network: A list of num_message_passing_rounds linear layers followed by ReLU activation functions
    #       to update the node states.
    # 4. Output network: A single layer which maps the node states to the latent space.

class GNNDecoder(torch.nn.Module): 
    """
    Graph Neural Network Decoder (Generative model)

    Keyword Arguments
    -----------------
        Z: (|V| x d)
    
    Returns
    -------
        A_hat: (|V| x |V|)
    # predict likelihood of all edges in graph
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.net = nn.Linear(self.latent_dim, 28*28)

    
    # simple dot product between latent variables
    def forward(self, Z):   
        # Z = Z.squeeze()
        # p_theta(A[u,v] = 1 | z_u, z_v) = sigmoid(z_u^T z_v) 
        print("Z shape ", Z.shape) # (1809,8)
        A_hat = torch.sigmoid(Z @ Z.T) # sigmoid of logits
        #A_hat = self.net(Z)
        print("A_hat shape ", A_hat.shape) # should be (1809, 1809) because 
        return A_hat

class VGAE(nn.Module):
    """
    Define a Variational Graph Autoencoder (VGAE) model.
    """
    def __init__(self, prior, encoder, decoder, k=1):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space. p(x|z)
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space. --> p(z|x)
        k: [int]
            The number of samples to use for IWAE. If set to 1, this is equivalent to using the ELBO loss.
            Note: Reparamterization trick is used and therefore estimates will vary for the same x.
        """
            
        super(VGAE, self).__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.k = k

    def calc_log_prob(self, Adj_pred, Adj, node_masks):
        """
        Function that calculates p(x|z) allowing for masking the calculated probabilities.
        """
        log_probs = Adj_pred.log_prob(Adj)
        masked_log_probs = log_probs[node_masks == 1]
        assert masked_log_probs.size(0) == torch.sum(node_masks), 'Number of unmasked nodes does not match the size of the masked log_probs!'
        return masked_log_probs.sum(dim=-1)
    
    def permute_adjacency_matrix(self, Adj_pred, perm):
        """
        Permutes an Adjacency matrix ensuring that it remains symmetric.
            Adj_pred: A tensor 
        """
        if Adj_pred.ndim == 3:
            # permute rows and columns to ensure permuted logits are symmetric
            Adj_pred = torch.index_select(Adj_pred, 1, perm)
            logits = torch.index_select(Adj_pred, 2, perm)
            # return as td.Bernoulli such that .log_prob method is defined
        else:
            # permute rows and columns to ensure permuted logits are symmetric
            Adj_pred = torch.index_select(Adj_pred, 0, perm)
            logits = torch.index_select(Adj_pred, 1, perm)

        return td.Bernoulli(logits=logits)

    def iwae(self, x, edge_index, batch):
        """
        Compute the IWAE for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        x = x.repeat(self.k, 1, 1)
        q = self.encoder(x, edge_index, batch)
        h = q.rsample()

        log_p_x_h = self.decoder(h).log_prob(x)
        log_p_h = self.prior().log_prob(h) 
        log_q_h_x = q.log_prob(h)
        marginal_likelihood = (log_p_x_h + log_p_h - log_q_h_x).view(self.k, x.size(0)//self.k)

        iwae = (torch.logsumexp(marginal_likelihood, dim=0) - torch.log(torch.tensor(self.k))).mean()
        
        return iwae

    def elbo(self, x, edge_index, batch):
        """
        Compute the ELBO for the given batch of data.
        L = sum_{G_i \in \mathbf{G}}} E_q(z|x)[log p(x|z)] - KL[q(z|x) || p(z)]
        # so for each graph, we want to max likelihood while min KL between posterior latent dist and prior
        
        In original paper: Optimize the variational lower bound wrt. W_i
        p_theta(G | Z) = prod_{u,v} p_theta(A_{u,v} =1 | z_u, z_v)

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        q = self.encoder(x, edge_index, batch)
        z = q.rsample() # reparameterization
        # assume independence between edges
        print("z shape ", z.shape) # (1809,8) 
        print("edge_index shape ", edge_index.shape) # (2, 4016)
        print("batch shape ", batch.shape) # (1809)
        # Predict the adjacency matrix from the latent variables
        A_hat = self.decoder(z)
        A = to_dense_adj(edge_index, batch).squeeze(0)
        print("Elbo A shape ", A.shape) # (100,28,28) which is (batch_size, max_num_nodes, max_num_nodes)
        print("Elbo A_hat shape ", A_hat.shape) 

        # Reconstruction error corresponds to a binary cross-entropy loss over the edge probabilities. T
        RE = torch.nn.functional.binary_cross_entropy(A_hat, A.float(), reduction='sum')
        print("RE shape ", RE.shape) # (100,28,28) which is (batch_size, max_num_nodes, max_num_nodes)
        # max_nodes = batch.max().item() + 1
        # elbo_loss = 0
        # for i in range(max_nodes):
        #     sub_mask = (batch == i)
        #     sub_edge_index = filter_edges(edge_index, sub_mask)
        #     sub_z = z[sub_mask]

        #     # Create the adjacency matrix for this subgraph
        #     sub_A_hat = self.decoder(sub_z)
        #     sub_A = to_dense_adj(sub_edge_index, max_num_nodes=sub_mask.sum().item()).squeeze(0)

        #     # Compute reconstruction error using binary cross-entropy
        #     sub_recon_loss = torch.nn.functional.binary_cross_entropy(sub_A_hat, sub_A.float(), reduction='sum')

        #     elbo_loss += sub_recon_loss

        # Average the ELBO loss across all graphs
        # elbo_loss /= max_nodes

        # A = to_dense_adj(edge_index, batch).squeeze(0)
        # A_dense, mask = to_dense_batch(x, batch, fill_value=0)
        # A = to_dense_adj(edge_index, batch, max_num_nodes=A_dense.size(1))


        
        # Calculate reconstruction error using binary cross-entropy
        #RE = torch.nn.functional.binary_cross_entropy(A_hat, A.float(), reduction='sum')

        KL = q.log_prob(z) - self.prior().log_prob(z)
        #KL = td.kl_divergence(q, self.prior()).sum()
        #elbo = (elbo_loss + KL)
        elbo = (RE + KL).mean()
        """
        Original code: 
        elbo_old = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        """

        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x, edge_index, batch):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x, edge_index, batch) if self.k == 1 else -self.iwae(x, edge_index, batch)
