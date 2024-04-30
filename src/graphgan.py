from functools import cached_property
import pdb
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.distributions as td
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch
import os

from src.utils import get_mutag_dataset, plot_adj
from src.gnn import GAN_MPNN, GraphConvNN

torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

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
    def __init__(self, 
                 state_dim: int, 
                 max_output_dim: int, 
                 num_hidden: int = 128,
                 num_layers: int = 3):
        super().__init__()
        state_dim_p_1 = state_dim + 1 # ! add extra NN connection for number of nodes conditioning
        # state_dim_p_1 = state_dim# + 1 # ! add extra NN connection for number of nodes conditioning
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_dim_p_1, num_hidden),
            torch.nn.ReLU(),
            *([torch.nn.Dropout1d(p=0.2), torch.nn.Linear(num_hidden, num_hidden),torch.nn.ReLU()] * num_layers),
            torch.nn.Linear(num_hidden, max_output_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Generator(torch.nn.Module):
    def __init__(self, gnn: GAN_MPNN|GraphConvNN|MultiLayerPerceptron, ndist: NDist, state_dim: int):
        super().__init__()
        self.gnn = gnn
        self.ndist = ndist
        self.state_dim = state_dim
        self.max_num_nodes = ndist.max_nodes
        self.sigmoid = torch.nn.Sigmoid()
        _loc, _scale = torch.zeros((self.state_dim,1)), torch.ones((self.state_dim,1)) # ? row or column vecs?
        self.seed_dist = td.Independent(td.Normal(loc=_loc,scale=_scale), 1)


    def compute_edge_dist(self, 
                        sample_shape: torch.Size):
        z = self.seed_dist.sample(sample_shape=sample_shape)
        ns = self.ndist.sample(sample_shape)

        # stack n and z to provide some conditioning for the generator # TODO ns currently commented out
        ns_conc = ns if ns.dim() == 2 else ns.unsqueeze(1)
        # inp = torch.hstack([z.squeeze(-1)])#, ns_conc])
        inp = torch.hstack([z.squeeze(-1), ns_conc])
        bernoulli_params = self.sigmoid(self.gnn(inp))
        # cont_dist for training, since we can backprop through it
        return ns, td.Independent(td.ContinuousBernoulli(bernoulli_params), 1)


    def edge_sample_to_batch_cont(self, ns, triu_edges):
        soft_adjs = []
        batches = []
        node_counter = 0
        max_num_nodes = ns.max()
        full_adj = torch.zeros((max_num_nodes,max_num_nodes)) # ! consistent indexing??
        for i,cur_n in enumerate(ns):
            adj = full_adj.clone()
            triu_idx = self.get_triu_idx(cur_n)
            triu_size = self.triu_size(cur_n)
            adj[triu_idx[0],triu_idx[1]] = triu_edges[i,:triu_size]
            adj[triu_idx[1],triu_idx[0]] = triu_edges[i,:triu_size]
            soft_adjs.append(adj.unsqueeze(0))
            batches.append(torch.tensor(cur_n*[i]))
            node_counter += cur_n
        
        x = self.static_x(node_counter)
        batched_soft_adj = torch.vstack(soft_adjs)
        batch = torch.hstack(batches)

        return x, batched_soft_adj, batch
    

    def sample(self, num_samples: int = 1):
        with torch.no_grad():
            self.gnn.eval() # disable dropout
            sample_shape = (num_samples,)
            ns, triu_edge_dist = self.compute_edge_dist(sample_shape) # compute edge prob dist
            triu_edges = triu_edge_dist.sample()

            x,batched_soft_adj,batch = self.edge_sample_to_batch_cont(ns, triu_edges)
            batched_edge_index = []
            node_counter = 0
            for i,soft_adj in enumerate(batched_soft_adj):
                edge_index = torch.nonzero((soft_adj >= .5)).t()
                batched_edge_index.append(edge_index + node_counter)
                node_counter += ns[i]
            batched_edge_index = torch.hstack(batched_edge_index)

            self.gnn.train() # reenable dropout
            return Batch(x=x, edge_index=batched_edge_index, batch=batch)


    def get_triu_idx(self, num_nodes: int):
        def compute_diag_idx(diag_size: int, full_size: int = None):
            full_size = full_size or diag_size # if full_size none, compute plain diag
            idx0 = torch.arange(diag_size).unsqueeze(0)
            idx1 = idx0 + (full_size - diag_size)
            return torch.vstack([idx0, idx1])

        triu_idx = torch.cat([
            compute_diag_idx(cur_n, num_nodes)
            for cur_n in range(1, num_nodes)
        ], dim=1)

        return triu_idx


    def triu_size(self, num_nodes: int):
        return int(num_nodes * (num_nodes-1) / 2)

    @cached_property
    def max_triu_size(self):
        return self.triu_size(self.ndist.max_nodes)

    @cached_property
    def full_triu_idx(self):
        return torch.triu_indices(self.max_num_nodes,self.max_num_nodes,1)
    
    def static_x(self, num_nodes: int):
        return torch.ones((num_nodes,7))

    def forward(self, num_samples: int = 1):
        sample_shape = (num_samples,)
        ns, triu_edge_dist = self.compute_edge_dist(sample_shape) # compute continuous edge prob dist
        triu_edges = triu_edge_dist.rsample() # reparametrization => differentiable operation
        
        x, batched_soft_adj, batch = self.edge_sample_to_batch_cont(ns, triu_edges)

        return x, batched_soft_adj, batch



class Discriminator(torch.nn.Module):
    def __init__(self, gnn: GAN_MPNN|GraphConvNN):
        super().__init__()
        self.gnn = gnn
        # nonlinearity = torch.nn.Sigmoid # ! NOTE: could be source of instability!
        nonlinearity = torch.nn.LeakyReLU # ! NOTE: could be source of instability!
        # nonlinearity = torch.nn.ReLU # ! NOTE: could be source of instability!
        # Project from state_dim to scalar and apply sigmoid => i.e. output probability
        self.score_model = torch.nn.Sequential(torch.nn.Linear(self.gnn.state_dim, 1), nonlinearity())

    def forward(self, graphs: tuple|Batch):
        if graphs.__class__.__name__ == 'DataBatch': # TODO give mode instead?
            static_x = torch.ones_like(graphs.x) # ! NOTE: Verify that this does not break stuff
            adjs = to_dense_adj(graphs.edge_index, graphs.batch)
            # TODO: evaluate this smoothing => they suck (seemingly)
            adjs *= (1. - torch.rand_like(adjs)*0.01) # ! smooth links
            adjs += (torch.rand_like(adjs)*0.01)      # ! smooth nonlinks
            adjs = torch.clamp(adjs, min=0, max=1)
            batch = graphs.batch
        else:
            static_x, adjs, batch = graphs # from generator
        state = self.gnn(static_x.double(), adjs, batch)
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


    def forward_old(self, graphs: Batch, disc: bool):
        m = graphs.batch.max().long().item() + 1
        
        gen_graphs = self.generator(num_samples=m)
        D_G_z = self.discriminator(gen_graphs)
        self.last_D_G_z = D_G_z

        if disc:
            D_x = self.discriminator(graphs)
            self.last_D_x = D_x
            disc_fake_loss = torch.log(1 - D_G_z).mean(dim=0)
            disc_real_loss = torch.log(D_x + self.eps).mean(dim=0)
            return -(disc_real_loss + disc_fake_loss)  # Maximize correct classification

        # use modified generator loss (https://developers.google.com/machine-learning/gan/loss)
        gen_loss = torch.log(D_G_z + self.eps).mean(dim=0)
        return -gen_loss  # Minimize the discriminator's ability to recognize generated data
    
    
    def forward(self, graphs: Batch, disc: bool):
        m = graphs.batch.max().long().item() + 1
        
        gen_graphs = self.generator(num_samples=m)
        D_G_z = self.discriminator(gen_graphs)
        self.last_D_G_z = D_G_z

        if disc:
            D_x = self.discriminator(graphs)
            self.last_D_x = D_x
            disc_real_loss = D_x.mean(dim=0)
            disc_fake_loss = D_G_z.mean(dim=0)
            return disc_fake_loss - disc_real_loss # critic loss (Wasserstein loss)

        gen_loss = D_G_z.mean(0)
        return -gen_loss  # Maximize the critic's function value for generated data



def train_gan(gan: GraphGAN, 
              dataloader: DataLoader,
              n_epochs: int,
              disc_lr: float, 
              gen_lr: float,
              disc_train_steps: int = 1,
              gen_train_steps: int = 1,
              ):
    sample_adj_from_gen = lambda: to_dense_adj(gan.generator.sample(1).edge_index).squeeze()

    # lambda to obtain new shuffled data loader each time it is called
    get_shuffled_data = lambda: iter(dataloader)

    # define optimizers for both discriminator and generator
    disc_optimizer = torch.optim.RMSprop(gan.discriminator.parameters(), lr=disc_lr)
    gen_optimizer =  torch.optim.RMSprop(gan.generator.parameters(),     lr=gen_lr)

    # combined_loss = torch.tensor(float('nan'))

    D_x_s = []
    D_G_z_s = []
    eval_freq = 25
    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            datas = get_shuffled_data() # TODO worth it to refactor not create new iterator every time?

            # Discriminator training steps
            for k in range(disc_train_steps):
                try: data = next(datas)
                except Exception: datas = get_shuffled_data(); data = next(datas)
                combined_loss = gan(data, disc = True)
                combined_loss.backward()
                disc_optimizer.step()
                # ! remove clipping if using non-Wasserstein GAN
                # ! Clip weights of discriminator 
                for p in gan.discriminator.parameters():
                    p.data.clamp_(-0.1, 0.1) # TODO find best clip value?
                    # p.data.clamp_(-0.01, 0.01) # TODO find best clip value?
                    # p.data.clamp_(-0.05, 0.05) # TODO find best clip value?
                disc_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            
            try: datas = get_shuffled_data(); data = next(datas)
            except Exception: datas = get_shuffled_data(); data = next(datas)
                
            # Generator training steps
            # gan.discriminator.requires_grad_ = False # disable gradients for discriminator
            for k in range(gen_train_steps):
                adv_loss = gan(data, disc = False) # we only use data for batch size 
                adv_loss.backward()
                gen_optimizer.step()
                gen_optimizer.zero_grad()
                disc_optimizer.zero_grad()
            # gan.discriminator.requires_grad_ = True # disable gradients for discriminator
            disc_optimizer.zero_grad()

            if (epoch % eval_freq == 0):
                mean_D_x = gan.last_D_x.mean().item()
                mean_D_G_z = gan.last_D_G_z.mean().item()
                pbar.set_description(f"[@ epoch {epoch}]: E[D(G(z))]={mean_D_G_z:.4f}, E[D(x)]={mean_D_x:.4f}")
                D_x_s.append(mean_D_x)
                D_G_z_s.append(mean_D_G_z)
                ts = torch.arange(len(D_x_s)) * eval_freq # ! hardcoded
                plt.plot(ts, D_x_s, label='E[D(x)]')
                plt.plot(ts, D_G_z_s, label='E[D(G(z))]')
                # plt.ylim(bottom=0., top=1.)
                plt.legend()
                plt.savefig("gan_training_curve.pdf")
                plt.close('all') # ? clean up

            if (epoch % 1_000 == 0) or (epoch == n_epochs-1): # make sample plots
                fig, axs = plt.subplots(3, 3, figsize=(18, 18))  # Create a grid of 3x3 for 3 rows and 3 columns
                for _i in range(3):
                    for _j in range(3):
                        plot_adj(sample_adj_from_gen(), axs[_i,_j])
                plt.savefig(f"samples/training/epoch_{epoch}.png")
                plt.close('all') # ? clean up



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model-dir', type=str, default="models", help='directory the model state dict is located in (default: %(default)s)')
    parser.add_argument('--model-state-dict', type=str, default="GraphGAN.pt", help='file to save model state dict to or load model state dict from (default: %(default)s)')
    # parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--disc-net', type=str, default='mpnn', choices=['mpnn', 'gcn'], help='which type of GNN to use for discriminator (default: %(default)s)')
    parser.add_argument('--disc-train-steps', type=int, default=1, help='number training steps for discriminator each round (default: %(default)s)')
    parser.add_argument('--gen-train-steps', type=int, default=1, help='number training steps for generator each round (default: %(default)s)')
    parser.add_argument('--mp-rounds', type=int, default=4, help='number of message passing rounds encoder network (default: %(default)s)')
    parser.add_argument('--filter-length', type=int, default=3, help='length of the GCN filter (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training (default: %(default)s)')
    parser.add_argument('--n-epochs', type=int, default=5_001, help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--state-dim', type=int, default=10, help='dimension of node state variables (default: %(default)s)')
    parser.add_argument('--num-hidden-gen-mlp', type=int, default=128, help='number of hidden units for each layer of the generator\'s MLP (default: %(default)s)')
    parser.add_argument('--num-layers-gen-mlp', type=int, default=5, help='number of hidden layers for each layer of the generator\'s MLP (default: %(default)s)')
    parser.add_argument('--gen-lr',  type=float, default=1e-5, help='learning rate for generator (default: %(default)s)')
    parser.add_argument('--disc-lr', type=float, default=1e-5, help='learning rate for discriminator (default: %(default)s)')
    args = parser.parse_args()

    print('\n# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
    print("")

    model_dir =             args.model_dir
    model_state_dict_path = args.model_state_dict

    if args.mode == 'train':
        """
        ================================================Train================================================
        """
        # ! hyperparams / training specific
        disc_net =               args.disc_net
        state_dim =              args.state_dim
        message_passing_rounds = args.mp_rounds
        filter_length =          args.filter_length # TODO try GCN ?
        n_epochs =               args.n_epochs
        batch_size =             args.batch_size
        disc_lr =                args.disc_lr
        gen_lr =                 args.gen_lr
        num_hidden_units =       args.num_hidden_gen_mlp
        num_layers =             args.num_layers_gen_mlp
        disc_train_steps =       args.disc_train_steps
        gen_train_steps =        args.gen_train_steps

        dataset = get_mutag_dataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        node_feature_dim = 7 # const

        # create distribution of node counts over the dataset
        ndist = NDist(dataset)
        max_num_nodes = ndist.max_nodes
        max_output_dim = int(max_num_nodes*(max_num_nodes-1)/2)

        # create generator and discriminator
        gen_net = MultiLayerPerceptron(state_dim=state_dim, 
                                       max_output_dim=max_output_dim, 
                                       num_hidden=num_hidden_units, 
                                       num_layers=num_layers)
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

        # Initialize GAN
        gan = GraphGAN(gen, disc)

        # fig, axs = plt.subplots(3, 3, figsize=(18, 18))  # Create a grid of 3x3 for 3 rows and 3 columns
        # # First col: real graph samples
        # for i in range(3):
        #     plot_adj(sample_real_adj(), axs[i, 0], name="Real Graph")
        # # Second col: generated graph samples before training
        # for i in range(3):
        #     plot_adj(sample_adj_from_gen(), axs[i, 1], name="GAN before training")

        # Train GAN
        train_gan(gan, dataloader, n_epochs=n_epochs, 
                  disc_lr=disc_lr, 
                  gen_lr=gen_lr, 
                  disc_train_steps=disc_train_steps,
                  gen_train_steps=gen_train_steps)
        # create folder if it does not exist
        os.makedirs(os.path.dirname(f'{model_dir}/{model_state_dict_path}'), exist_ok=True)
        torch.save(gan.state_dict(), f'{model_dir}/{model_state_dict_path}')

        # Third col: generated graph samples after training
        # for i in range(3):
        #     plot_adj(sample_adj_from_gen(), axs[i, 2], name=f"GAN after training {n_epochs} epochs")

        plt.savefig("samples/comparison.png")  # Save the complete figure to file
        # plt.show()
        print("\ntraining done! Exiting.\n")


    elif args.mode == 'sample':
        """
        ========================================Generate and Visualize========================================
        """
        torch.load(f'{model_dir}/{model_state_dict_path}')

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # plot_adj(sample_adj(), axs[0], name="Sampled Graph")
        # plot_adj(sample_model(baseline_model), axs[1], name="Generated Graph")

        plt.show()