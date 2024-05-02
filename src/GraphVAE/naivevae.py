import torch
import torch.nn as nn
import torch.distributions as td
from torch_geometric.utils import to_dense_adj
import pdb

"""
A very simple VAE implementation not using masking or "permutation invariant" reconstruction term.
This is not used in the final report only for comparison purposes.
"""

class NaiveVAE(nn.Module):
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
        super(NaiveVAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.ndist = ndist

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
        RE =  self.decoder(z).log_prob(Adj)
        
        RE = torch.mean(torch.sort(RE, dim=-1, descending=True)[0][:int(Adj.size(0) * 0.05)])
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

    def sample(self, n_samples=1, return_probs=False):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        samples = []
        mu_probs = torch.zeros(1, 28, 28)
        for i in range(n_samples):
            n_nodes = self.ndist.sample_N((1,)) # sample number of nodes
            # sample full max_nodes x max_nodes A matrix
            z = self.prior().sample(torch.Size([1]))
            decoder_sample = self.decoder(z).sample()
            mu_probs += self.decoder(z).base_dist.probs
            upper = torch.triu(decoder_sample, diagonal=1)
            Adj_sample = upper + upper.transpose(1, 2)
            # downsample A
            Adj_downsampled = self.mask_sample(Adj_sample, n_nodes[0])
            # store
            samples.append(Adj_downsampled)
        
        mu_probs /= n_samples
        if return_probs:
            return samples, mu_probs
        else:
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