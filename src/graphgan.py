from functools import cached_property
from tqdm import tqdm

import torch
import torch.distributions as td
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

from src.gnn import GanMPN, GraphConvNN
from src.utils import get_mutag_dataset


class NDist:
    """
    Wrap empirical distribution of number of nodes in graphs.
    """
    def __init__(self, dataset):
        # ! load all graphs in dataset for computing graph stats
        dataloader = DataLoader(dataset, batch_size=len(dataset))

        graphs = next(iter(dataloader))
        num_graphs = len(graphs)

        # compute empirical distribution of N, i.e. #nodes
        graph_idxs = torch.arange(num_graphs).unsqueeze(1)
        mask = graphs.batch == graph_idxs
        nodes_per_graph = (mask).sum(1) # how many nodes are in each graph
        N_dist = torch.vstack(torch.unique(nodes_per_graph, return_counts=True))
        N_dist = N_dist.float()

        N_dist[1] /= num_graphs # finally, convert counts to probabilities (we needed to use it to compute probs_given_N)
        self._N_dist = N_dist
    
    @cached_property
    def max_nodes(self):
        return self._N_dist[0].max().int().item()

    def sample(self, sample_shape: torch.Size):
        idx = torch.multinomial(self._N_dist[1], num_samples=sample_shape[0])
        return self._N_dist[0, idx].int()


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, state_dim: int, max_output_dim: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, max_output_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Generator(torch.nn.Module):
    # TODO: outputs (potentially padded) max size graph

    def __init__(self, gnn: GanMPN|GraphConvNN|MultiLayerPerceptron, ndist: NDist, state_dim: int = 16):
        super().__init__()
        self.gnn = gnn
        self.ndist = ndist
        self.state_dim = state_dim
        self.sigmoid = torch.nn.Sigmoid()
        _loc, _scale = torch.zeros((self.state_dim,1)), torch.ones((self.state_dim,1)) # ? row or column vecs?
        self.seed_dist = td.Independent(td.Normal(loc=_loc,scale=_scale), 1)


    def sample_adj_dist(self, sample_shape: torch.Size):
        z = self.seed_dist.sample(sample_shape=sample_shape)
        ns = self.ndist.sample(sample_shape)
        # TODO stack n to z and update gnn to support it 
        bernoulli_params = self.sigmoid(self.gnn(z.squeeze()))
        return ns, td.Independent(td.Bernoulli(bernoulli_params), 1)
  
    def sample(self, num_samples: int = 1):
        with torch.no_grad():
            sample_shape = (num_samples,)
            ns, adj_params = self.sample_adj_dist(sample_shape)
            triu_indices = adj_params.sample().long()
            # TODO convert to edge_index / adj matrix
            # use n for masking
            # return list?
            ...

    def backprop_through_sample_op(self):
        # manually allow gradients to flow back through the Bernoulli sample operation using "pass-through" operator
        self._last_probs_for_grad.backward(gradient=self._last_triu_samples)

    @cached_property
    def triu_idx(self):
        return torch.triu_indices(max_num_nodes,max_num_nodes,1)
    
    # @cached_property
    def static_x(self, num_nodes: int):
        return torch.ones((num_nodes,7))

    def forward(self, num_samples: int = 1):
        sample_shape = (num_samples,)
        ns, triu_edge_dist = self.sample_adj_dist(sample_shape)
        triu_edges = triu_edge_dist.sample()#.long()
        
        triu_edges.requires_grad = True # needed for passing backprop through Bernoulli sample operation 

        batch_size = num_samples
        edge_indices = []
        batches = []
        node_counter = 0
        for i in range(batch_size):
            edge_index = self.triu_idx[:,triu_edges[i].bool()]
            # TODO TODO TODO: Right now we generate full 28x28 matrices. We should mask out top n rows and columns
            # concat with reverse edge index
            edge_index = torch.hstack([edge_index, torch.vstack([edge_index[1], edge_index[0]])])
            edge_indices.append(edge_index + node_counter)
            batches.append(torch.tensor(ns[i]*[i]))
            node_counter += ns[i]
        batched_edge_index = torch.hstack(edge_indices)
        batch = torch.hstack(batches)

        # save distribution parametrization for backprop
        self._last_triu_samples = triu_edges
        self._last_probs_for_grad = triu_edge_dist.base_dist.probs 

        x = self.static_x(node_counter)
        return Batch(x=x, edge_index=batched_edge_index, batch=batch)




        
class Discriminator(torch.nn.Module):
    def __init__(self, gnn: GanMPN|GraphConvNN):
        super().__init__()
        self.gnn = gnn
        # TODO: linear layer to output scalar
        bernoulli_linkage = torch.nn.Sigmoid # ! NOTE: could be source of instability! 
        # Project from state_dim to scalar and apply sigmoid
        self.score_model = torch.nn.Sequential(torch.nn.Linear(self.gnn.state_dim, 1), bernoulli_linkage())

    def forward(self, graphs: Batch):
        x_zeros = torch.ones_like(graphs.x) # ! NOTE: Verify that this does not break stuff
        state = self.gnn(x_zeros, graphs.edge_index, graphs.batch)
        return self.score_model(state)


class GraphGAN(torch.nn.Module):
    def __init__(self,
                 gen_net: Generator,
                 disc_net: Discriminator):
        super().__init__()
        self.generator = gen_net
        self.discriminator = disc_net
        self.state_dim = self.generator.state_dim

    def forward(self, graphs: Batch):
        """
        @params:
            x: Graph - batched real input data.
        """
        m = graphs.batch.max().long().item() + 1 # number of samples for both real and fake samples

        d_x = self.discriminator(graphs) # prob of real data
        
        d_g_z = self.discriminator(self.generator(num_samples=m)) # prob of generated data

        disc_loss = torch.log(1 - d_x).mean(dim=0) # loss for discriminator on real data
        adv_loss = torch.log(d_g_z).mean(dim=0)    # loss for discriminator vs generator

        return disc_loss + adv_loss



def train_gan(gan: GraphGAN, 
              dataloader: DataLoader,
              n_epochs: int = 1_000,
              disc_train_steps: int = 1,
              ):

    # lambda to obtain new shuffled data loader each time it is called
    get_shuffled_data = lambda: iter(dataloader)

    # define optimizers for both discriminator and generator
    disc_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=1e-3)
    gen_optimizer = torch.optim.Adam(gan.generator.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        datas = get_shuffled_data()

        for k in range(disc_train_steps):
            data = next(datas)
            loss = gan(data)
            loss.backward()
            disc_optimizer.step()
            disc_optimizer.zero_grad()
        
        data = next(datas)
        loss = gan(data)
        loss.backward()
        gan.generator.backprop_through_sample_op() # manually handle backprop for the generator through Bernoulli sample
        gen_optimizer.step()
        gen_optimizer.zero_grad()





if __name__ == '__main__':

    dataset = get_mutag_dataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # get biggest node count
    ndist = NDist(dataset)
    max_num_nodes = ndist.max_nodes
    max_output_dim = int(max_num_nodes*(max_num_nodes-1)/2)

    state_dim = 16
    gennet = MultiLayerPerceptron(state_dim=state_dim, max_output_dim=max_output_dim)
    gen = Generator(gennet, ndist=ndist, state_dim=state_dim)
    batch = gen(3)
    mpnn = GanMPN(node_feature_dim=7, state_dim=state_dim, num_message_passing_rounds=5)
    disc = Discriminator(mpnn)

    gan = GraphGAN(gen, disc)



    batch_size = 3

    # ns, samples = gen.sample(batch_size)

    # idx = torch.triu_indices(max_num_nodes,max_num_nodes,1)

    # mask triu indices with samples to get edge_index

    # edge_indices = []
    # batches = []
    # for i in range(batch_size):
    #     edge_index = idx[:,samples[i].bool()]
    #     # concat with reverse edge index
    #     edge_index = torch.hstack([edge_index, torch.vstack([edge_index[1],edge_index[0]])])
    #     edge_indices.append(edge_index)
    #     batches.append(torch.tensor(ns[i]*[i]))
    # batched_edge_index = torch.hstack(edge_indices)
    # batched_batch = torch.hstack(batches)



    # Convert to dense adj
    # dense_adj = torch.zeros((max_num_nodes,max_num_nodes))
    # dense_adj[edge_index[0],edge_index[1]] = 1

    
    # adjs = torch.zeros((2,max_num_nodes,max_num_nodes)).long()
    # adjs[:,idx[0],idx[1]] = samples
    # adjs[:,idx[1],idx[0]] = samples

    # print(torch.all(adjs[0] == dense_adj)) # should be True

    # import matplotlib.pyplot as plt


    # # plot adjacencies side by side
    # fig, axs = plt.subplots(1,2)
    # axs[0].spy(adjs[0])
    # axs[1].spy(dense_adj)
    # plt.show()

        

    batch = next(iter(dataloader))
    disc(batch)


    train_gan(gan, dataloader)

    # torch.save(gan.state_dict(), 'models/gan.pt')


    
    # Make dummy example of upper triangular edges 
    # torch_test = torch.tensor([[0, 1, 0, 0, 0, 0],[1, 1, 0, 0, 0, 0]])
    torch_test = torch.tensor([0, 1, 0, 0, 0, 0])


    # num_tri = n**2/2-n
    

    t = torch_test.size(0)
    n = int((1 + (1 + 8*t)**0.5)/2)

    adj = torch.zeros((n,n)).long()
    triu_indices = torch.triu_indices(n,n,1)
    adj[triu_indices[0], triu_indices[1]] = torch_test







    # convert torch test to edge index  

    

