import torch
import torch.nn as nn
import torch.distributions as td
from torch_geometric.utils import to_dense_adj
import pdb

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, ndist):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space. p(x|z)
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space. p(z|x)
        n_dist: NDist
                Empirical distribution over the number of nodes.
        """ 
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.ndist = ndist 
    
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

    def simple_reconstruction_with_perm(self, z, Adj, node_masks):
        """
        Calculates n_perms random permutations of the predicted adjacency matrix. 
        Returns: The lowest reconstruction error.
        """

        Adj_pred = self.decoder(z).base_dist # predict adjacency matrix 
        n_perms = 30
        rec_errors = torch.empty((Adj.size(0), n_perms))
        
        for i in range(Adj.size(0)): # loop over number of permutations to try
            for j in range(n_perms): # loop over batch
                perm = torch.randperm(Adj.size(2)) # sample random permutation
                Adj_perm = self.permute_adjacency_matrix(Adj_pred.logits[j], perm) # permute predicted A
                rec_error = self.calc_log_prob(Adj_perm, Adj[j], node_masks[j]) # calculate reconstruction error
                # store permutation and reconstruction error

                rec_errors[i, j] = rec_error
        
        return torch.sum(torch.max(rec_errors, dim=1).values)

    def elbo(self, x, edge_index, batch, Adj, node_masks):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
            x:
            edge_index:
            batch:
            Adj: A 3D tensor of size batch_size x max_num_nodes x max_num_nodes
            node_masks: A 3D tensor of the same size as Adj 
        """
        q = self.encoder(x, edge_index, batch)
        z = q.rsample() # reparameterization
        
        RE = self.simple_reconstruction_with_perm(z, Adj, node_masks)
        # RE = self.calc_log_prob(z, Adj, node_masks) #self.decoder(z)[node_masks].log_prob(Adj[node_masks])
        KL = q.log_prob(z) - self.prior().log_prob(z)

        elbo = (RE - KL).mean()
        """
        Original code: 
        elbo_old = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        """
        return elbo
    
    def forward(self, x, edge_index, batch, Adj, node_masks):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:

        """
        return -self.elbo(x, edge_index, batch, Adj, node_masks)

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        samples = []
        for i in range(n_samples):
            n_nodes = self.ndist.sample((1,)) # sample number of nodes
            
            # sample full max_nodes x max_nodes A matrix
            z = self.prior().sample(torch.Size([1]))
            decoder_sample = self.decoder(z).sample()
            upper = torch.triu(decoder_sample, diagonal=1)
            Adj_sample = upper + upper.transpose(1, 2)
            # downsample A
            Adj_downsampled = self.mask_sample(Adj_sample, n_nodes[0])
            # store
            samples.append(Adj_downsampled)

        return samples

    def mask_sample(self, Adj_sample, n_nodes: int):
        mask = torch.zeros_like(Adj_sample)
        mask[:, :n_nodes, :n_nodes] = 1
        return Adj_sample[mask == 1].view(n_nodes, n_nodes)
    
class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M, dtype=torch.float32), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M, dtype=torch.float32), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
    
class GaussianEncoder(nn.Module):
    def __init__(self, mu_encoder_net, sigma_encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.mu_encoder_net = mu_encoder_net
        self.sigma_encoder_net = sigma_encoder_net

    def forward(self, x, edge_index, batch):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)

        Returns 
        """
        mean = self.mu_encoder_net(x, edge_index, batch)
        std = self.sigma_encoder_net(x, edge_index, batch)
        
        # NOTE: a small number is added to avoid error
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)+1e-6), 1)
        
class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
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
        return td.Independent(td.Bernoulli(logits=logits), 2)