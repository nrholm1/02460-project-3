from functools import cached_property
import pdb
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.distributions as td
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj

from src.utils import get_mutag_dataset, plot_adj
from src.gnn import GAN_MPNN, GraphConvNN


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
        idx = torch.multinomial(self._N_dist[1], num_samples=sample_shape[0], replacement=True)
        return self._N_dist[0, idx].int()


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, state_dim: int, max_output_dim: int, num_hidden = 128):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_dim, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, max_output_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Generator(torch.nn.Module):
    # TODO: outputs (potentially padded) max size graph

    def __init__(self, gnn: GAN_MPNN|GraphConvNN|MultiLayerPerceptron, ndist: NDist, state_dim: int):
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
        bernoulli_params = self.sigmoid(self.gnn(z.squeeze(-1)))
        return ns, td.Independent(td.Bernoulli(bernoulli_params), 1)


    def sample_to_batch(self, ns, triu_edges):
        edge_indices = []
        batches = []
        node_counter = 0
        for i,cur_n in enumerate(ns):
            # Full 28 node graphs are generated (max size in empirical dist.), then masked
            edge_index = self.triu_idx[:,triu_edges[i].bool()]
            mask = torch.all(edge_index < cur_n, dim=0) # mask out top n rows and columns
            edge_index = edge_index[:,mask]
            edge_index = torch.hstack([edge_index, torch.vstack([edge_index[1], edge_index[0]])]) # make symmetric
            edge_indices.append(edge_index + node_counter)
            batches.append(torch.tensor(cur_n*[i]))
            node_counter += cur_n
        
        x = self.static_x(node_counter)
        batched_edge_index = torch.hstack(edge_indices)
        batch = torch.hstack(batches)

        return x ,batched_edge_index, batch


    def sample(self, num_samples: int = 1):
        with torch.no_grad():
            sample_shape = (num_samples,)
            ns, triu_edge_dist = self.sample_adj_dist(sample_shape)
            triu_edges = triu_edge_dist.sample()#.long()

            x,batched_edge_index,batch = self.sample_to_batch(ns, triu_edges)

            return Batch(x=x, edge_index=batched_edge_index, batch=batch)


    def backprop_through_sample_op(self, grad):
        """ Manually pass gradients straight through the non-differentiable Bernoulli sample operation. """
        last_m = self._last_batch.max()+1
        acc_grads = torch.zeros((last_m, self.triu_size))
        for i in range(last_m): 
            acc_grad = grad[self._last_batch == i].mean(dim=1)
            acc_grads[i,:len(acc_grad)] = acc_grad
        self._last_probs_for_grad_pass.backward(gradient=acc_grads)

    @cached_property
    def triu_size(self):
        return int(self.ndist.max_nodes * (self.ndist.max_nodes-1) / 2)

    @cached_property
    def triu_idx(self):
        return torch.triu_indices(max_num_nodes,max_num_nodes,1)
    
    def static_x(self, num_nodes: int):
        return torch.ones((num_nodes,7))

    def forward(self, num_samples: int = 1):
        sample_shape = (num_samples,)
        ns, triu_edge_dist = self.sample_adj_dist(sample_shape)
        triu_edges = triu_edge_dist.sample()#.long()
        
        x,batched_edge_index,batch = self.sample_to_batch(ns, triu_edges)

        # save distribution parametrization for manual backprop at discontinuity (sample operation)
        self._last_probs_for_grad_pass = triu_edge_dist.base_dist.probs
        # save batch (node indexing) to know which grads corresponds to the correct graph sample
        self._last_batch = batch

        return Batch(x=x, edge_index=batched_edge_index, batch=batch)



class Discriminator(torch.nn.Module):
    def __init__(self, gnn: GAN_MPNN|GraphConvNN):
        super().__init__()
        self.gnn = gnn
        bernoulli_linkage = torch.nn.Sigmoid # ! NOTE: could be source of instability! 
        # Project from state_dim to scalar and apply sigmoid => i.e. output probability
        self.score_model = torch.nn.Sequential(torch.nn.Linear(self.gnn.state_dim, 1), bernoulli_linkage())

    def get_input_grad(self):
        return self.gnn.get_input_grad()

    def forward(self, graphs: Batch):
        static_x = torch.ones_like(graphs.x) # ! NOTE: Verify that this does not break stuff
        state = self.gnn(static_x, graphs.edge_index, graphs.batch)
        return self.score_model(state)



class GraphGAN(torch.nn.Module):
    def __init__(self,
                 gen_net: Generator,
                 disc_net: Discriminator,
                 eps: float = 1e-9):
        super().__init__()
        self.generator = gen_net
        self.discriminator = disc_net
        self.state_dim = self.generator.state_dim
        self.eps = eps # ! add to log terms to counter -inf's and for numerical stability

    def backprop_through_discontinuity(self):
        self.generator.backprop_through_sample_op(grad=self.discriminator.get_input_grad())

    def forward(self, graphs: Batch, disc: bool):
        """
        @params:
            x: Graph - batched real input data.
            disc: bool - flag for loss mode, i.e. for discriminator or generator
        """
        m = graphs.batch.max().long().item() + 1 # number of samples for both real and fake samples

        # for both training modes, we need to compute adversarial loss
        D_G_z = self.discriminator(self.generator(num_samples=m))      # prob that generated data is fake D(G(z))
        adv_loss = torch.log(D_G_z + self.eps).mean(dim=0)             # loss for discriminator vs generator

        if disc: # if training discriminator, we need to optimize on real data as well
            D_x = self.discriminator(graphs)                           # prob that real data D(x) is fake
            disc_real_loss = torch.log(1 - D_x + self.eps).mean(dim=0) # loss for discriminator on real data
            return - (disc_real_loss + adv_loss) # ! return negated, since we want to maximize this?
        
        return adv_loss # else return just adversarial loss



def train_gan(gan: GraphGAN, 
              dataloader: DataLoader,
              n_epochs: int = 1_000,
              disc_train_steps: int = 5,
              ):
    sample_adj_from_gen = lambda: to_dense_adj(gan.generator.sample(1).edge_index).squeeze()

    # lambda to obtain new shuffled data loader each time it is called
    get_shuffled_data = lambda: iter(dataloader)

    # define optimizers for both discriminator and generator
    disc_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=1e-3)
    gen_optimizer =  torch.optim.Adam(gan.generator.parameters(),     lr=1e-3)

    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            datas = get_shuffled_data() # TODO worth it to refactor not create new iterator every time?

            gan.generator.requires_grad_ = False # disable gradients for generator
            for k in range(disc_train_steps):
                data = next(datas)
                combined_loss = gan(data, disc = True)
                combined_loss.backward()
                disc_optimizer.step()
                disc_optimizer.zero_grad()
            gan.generator.requires_grad_ = True # reenable gradients for generator
            # gen_optimizer.zero_grad()
                
            # Generator training step
            # data = next(datas)
            adv_loss = gan(data, disc = False) # reuse old data, since we only use the batch_size when disc==False
            adv_loss.backward()
            gan.backprop_through_discontinuity() # manually handle backprop for the generator through Bernoulli sample
            gen_optimizer.step()
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            pbar.set_description(f"Adv-loss: {adv_loss.item():.5e}, Combined-loss: {combined_loss.item():.5e}")

            # if epoch % 50 == 0 and epoch != 0:
            #     plot_adj(sample_adj_from_gen())
            #     plt.savefig(f"samples/epoch_{epoch}.png")
            #     plt.clf()




if __name__ == '__main__':
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    # parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    # parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    # parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='batch size for training (default: %(default)s)')
    # parser.add_argument('--n-epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: %(default)s)')
    # parser.add_argument('--mp-rounds', type=int, default=5, metavar='N', help='Number of message passing rounds encoder network (default: %(default)s)')
    # parser.add_argument('--state-dim', type=int, default=16, metavar='N', help='dimension of latent variable (default: %(default)s)')
    # args = parser.parse_args()

    model_dir =             "models"      # TODO: add to argparse
    model_state_dict_path = "GraphGAN.pt" # TODO: add to argparse

    """
    ================================================Train================================================
    """
    # ! hyperparams / training specific
    disc_net =               'mpnn' # TODO: add to argparse choices=[mpnn, gcn]
    state_dim =              8     # TODO: add to argparse
    message_passing_rounds = 3      # TODO: add to argparse
    filter_length =          3      # TODO: add to argparse
    n_epochs =               5_001  # TODO: add to argparse
    batch_size =             16     # TODO: add to argparse

    dataset = get_mutag_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    node_feature_dim = 7 # const


    # create distribution of node counts over the dataset
    ndist = NDist(dataset)
    max_num_nodes = ndist.max_nodes
    max_output_dim = int(max_num_nodes*(max_num_nodes-1)/2)

    # create generator and discriminator
    gen_net = MultiLayerPerceptron(state_dim=state_dim, max_output_dim=max_output_dim)
    gen = Generator(gen_net, ndist=ndist, state_dim=state_dim)

    # ! util function to sample 1 plot-ready adj matrix from the generator and dataset
    sample_adj_from_gen = lambda: to_dense_adj(gen.sample(1).edge_index).squeeze()
    dataloader_single = DataLoader(dataset, batch_size=1, shuffle=True) # dataloader for sampling single graphs
    sample_real_adj = lambda: to_dense_adj(next(iter(dataloader_single)).edge_index).squeeze() # sample and convert to dense adjacency matrix

    if disc_net == 'mpnn':
        disc_net = GAN_MPNN(node_feature_dim=node_feature_dim, 
                            state_dim=state_dim, 
                            num_message_passing_rounds=message_passing_rounds)
    elif disc_net == 'gcn':
        disc_net = GraphConvNN(node_feature_dim=node_feature_dim, 
                               filter_length=filter_length)
    disc = Discriminator(disc_net)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # plot sampled real graph
    plot_adj(sample_real_adj(), axs[0], name="Real Graph")

    # plot generated graph before training
    plot_adj(sample_adj_from_gen(), axs[1], name="GAN before training")
    
    # plot_adj(sample_adj_from_gen(), name="GAN before training")
    # plt.savefig("samples/before.png")
    # plt.clf()

    # create and train GAN
    gan = GraphGAN(gen, disc)
    train_gan(gan, dataloader, n_epochs=n_epochs)
    torch.save(gan.state_dict(), f'{model_dir}/{model_state_dict_path}')

    # plot generated graph after training
    plot_adj(sample_adj_from_gen(), axs[2], name=f"GAN after training {n_epochs} epochs")

    plt.savefig("samples/comparison.png")
    # plt.show()


    print("\ntraining done! Exiting.\n")
    exit()

    """
    ========================================Generate and Visualize========================================
    """
    torch.load(f'{model_dir}/{model_state_dict_path}')

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # plot_adj(sample_adj(), axs[0], name="Sampled Graph")
    # plot_adj(sample_model(baseline_model), axs[1], name="Generated Graph")

    plt.show()