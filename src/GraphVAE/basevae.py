import torch
import torch.nn as nn
import torch.distributions as td
from torch_geometric.utils import to_dense_adj
import pdb
from torch.nn.functional import binary_cross_entropy_with_logits

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
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

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

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, k=1):
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
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.k = k
        
    def iwae(self, x):
        """
        Compute the IWAE for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        x = x.repeat(self.k, 1, 1)
        q = self.encoder(x)
        h = q.rsample()

        log_p_x_h = self.decoder(h).log_prob(x)
        log_p_h = self.prior().log_prob(h) 
        log_q_h_x = q.log_prob(h)
        marginal_likelihood = (log_p_x_h + log_p_h - log_q_h_x).view(self.k, x.size(0)//self.k)
        iwae = (torch.logsumexp(marginal_likelihood, dim=0) - torch.log(torch.tensor(self.k))).mean()
        return iwae
    
    def calc_log_prob(self, z, Adj, node_masks):
        logits = self.decoder(z)
        log_probs = logits.base_dist.log_prob(Adj)
        return log_probs[node_masks].view(-1, Adj.size(0), Adj.size(1)).sum(dim=-1)

    def elbo(self, x, edge_index, batch):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        q = self.encoder(x, edge_index, batch)
        z = q.rsample() # reparameterization


        ### Get Adjacency matrices and node masks
        num_nodes_per_graph = torch.unique(batch, return_counts=True)[1]
        max_num_nodes = num_nodes_per_graph.max().item()

        Adj = to_dense_adj(edge_index, batch, max_num_nodes=max_num_nodes,
                           batch_size=batch.max().item()+1)
        node_masks = torch.zeros_like(Adj)
        
        for i, m in enumerate(num_nodes_per_graph):
            node_masks[i, :m, :m] = 1
        
        node_masks = node_masks.to(torch.long)

        RE = self.calc_log_prob(z, Adj, node_masks) #self.decoder(z)[node_masks].log_prob(Adj[node_masks])
        KL = q.log_prob(z) - self.prior().log_prob(z)

        elbo = (RE - KL).mean()
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
        #  torch.triu(z, diagonal=1)
        
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x, edge_index, batch):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x, edge_index, batch) if self.k == 1 else -self.iwae(x)