import torch
import torch.distributions as td

from src.gnn import MessagePassingNN, GraphConvNN


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self):
        ...


class GeneratorNet(torch.nn.Module):
    # TODO: outputs (potentially padded) max size graph

    def __init__(self, gnn: MessagePassingNN|GraphConvNN|MultiLayerPerceptron):
        super().__init__()
        self.gnn = gnn
        self.sigmoid = torch.nn.Sigmoid()
        _loc, _scale = torch.zeros((self.latent_dim,1)), torch.ones((self.latent_dim,1)) # ? row or column vecs?
        self.seed_dist = td.Independent(td.Normal(loc=_loc,scale=_scale), 1)
    
    def sample(self, sample_shape: torch.Size):
        z = self.seed_dist.rsample(sample_shape=sample_shape) # sample batch from seed dist
        bernoulli_params = self.sigmoid(self.gnn(z))
        return td.Independent(td.ContinuousBernoulli(bernoulli_params), 2).rsample()
    
    def forward(self, sample_shape: torch.Size):
        triu_edges = self.sample(sample_shape)

        


class DiscriminatorNet(torch.nn.Module):
    def __init__(self, gnn: MessagePassingNN|GraphConvNN):
        super().__init__()
        self.gnn = gnn
        # TODO: linear layer to output scalar
        bernoulli_linkage = torch.nn.Sigmoid # ! NOTE: could be source of instability! 
        self.score_model = torch.nn.Sequential(torch.nn.Linear(), bernoulli_linkage())
        ...

    def forward(self, x: torch.Tensor):
        state = self.gnn(x)
        return self.score_model(state)


class GraphGAN(torch.nn.Module):
    def __init__(self,
                 gen_net: GeneratorNet,
                 disc_net: DiscriminatorNet):
        super().__init__()
        self.generator = gen_net
        self.discriminator = disc_net
        self.latent_dim = self.generator.latent_dim

    def forward(self, x: torch.Tensor):
        """
        @params:
            x: torch.Tensor - batched real input data.
        """
        m = x.shape[0] # number of samples for both real and fake samples

        d_x = self.discriminator(x) # prob of real data
        
        d_g_z = self.discriminator(self.generator.sample((m,))) # prob of generated data

        disc_loss = torch.log(1 - d_x).mean(dim=0) # loss for discriminator on real data
        adv_loss = torch.log(d_g_z).mean(dim=0)    # loss for discriminator vs generator

        return disc_loss, adv_loss



def train_gan(gan: GraphGAN, 
              n_epochs: int = 1_000,
              disc_train_steps: int = 5,
              ):
    batch = ... # TODO
    x = ...
    disc_loss, adv_loss = gan(x)



if __name__ == '__main__':
    gen = GeneratorNet()
    disc = DiscriminatorNet()
    gan = GraphGAN()

    train_gan(gan)

    torch.save(gan.state_dict(), 'models/gan.pt')
