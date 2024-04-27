import argparse
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from src.GraphVAE.graphvae import build_model
from src.vae_trainer import VAETrainer
from src.utils import get_mutag_dataset
from src.vae_trainer import get_inputs
import pdb
from torch_geometric.utils import to_dense_adj
from src.graph_statistics import GraphStatistics
from src.baseline import Baseline
from src.graphgan import NDist


def simple_permute_example():
    Adj_true = torch.zeros((5, 5))
    mask_true = torch.zeros((5, 5))
    mask_true[:3, :3] = 1
    Adj_true[0, 1] = 1
    Adj_true[1, 0] = 1
    Adj_true[2, 1] = 1
    Adj_true[1, 2] = 1

    Adj_pred = torch.zeros((5, 5))
    Adj_pred[3, 1] = 1
    Adj_pred[1, 3] = 1
    Adj_pred[2, 4] = 1
    Adj_pred[4, 2] = 1

    perm = torch.randperm(5)
    Adj_perm = torch.index_select(Adj_pred, 0, perm)
    Adj_perm = torch.index_select(Adj_perm, 1, perm)
    
    pdb.set_trace()


def evaluate(model, dataloader, device):
    model.eval()

    _x = next(iter(dataloader))
    Adj, node_masks = get_inputs(_x.batch, _x.edge_index, model.ndist.max_nodes)

    data = next(iter(dataloader))
    data = data.to(device)
    loss = model(data.x, data.edge_index, data.batch, Adj, node_masks)

    print(f'Finished evaluation!\nneg. ELBO: {loss.item():.4f}')

if __name__ == '__main__':
    # Parse arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval', 'sample_posterior'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--n-rounds', type=int, default=25, metavar='N', help='Number of message passing rounds encoder network (default: %(default)s)')
    parser.add_argument('--embedding-dim', type=int, default=16, metavar='N', help='embedding dimension of encoder network (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=16, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='Standard_Normal', choices=['Standard_Normal', 'MoG', 'Flow', 'Vamp'], help='Type of prior distribution over latents e.g. p(z)')
    parser.add_argument('--k', type=int, default=1, help='The sample size when using IWAE loss (default: %(default)s)')
    args = parser.parse_args()
    
    # below values are hard coded since we only consider MUTAG
    node_feature_dim = 7 
    max_num_nodes = 28
    min_num_nodes = 10

    MUTAG_dataset = get_mutag_dataset(args.device)
    
    # Split into training and validation (same split as in exercise 10 except only test)
    rng = torch.Generator().manual_seed(0)
    train_dataset, test_dataset = random_split(MUTAG_dataset, (100, 88), generator=rng)
    # Create dataloader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=100)
    test_loader = DataLoader(test_dataset, batch_size=88)

    if args.mode == 'train':
        model = build_model(node_feature_dim, args.embedding_dim, args.n_rounds, args.latent_dim, MUTAG_dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        trainer = VAETrainer(model, optimizer, train_loader, args.epochs, args.device)
        trainer.train()
    
    elif args.mode == 'eval':
        ### Evaluate negative ELBO
        model = build_model(node_feature_dim, args.embedding_dim, args.n_rounds, args.latent_dim, MUTAG_dataset)
        # model.load_state_dict(torch.load('VAE_weights.pt', map_location=args.device))
        
        evaluate(model, test_loader, device=args.device)

        ### Evaluate uniqueness and novelty
        N_samples_novelty = 1000
        num_nodes_distribution = NDist(MUTAG_dataset)

        
        # model = build_model(node_feature_dim, args.embedding_dim, args.n_rounds, args.latent_dim, max_num_nodes)
        # model.load_state_dict(torch.load('VAE_weights.pt', map_location=args.device))
        
        VAE_samples = model.sample(N_samples_novelty)
        pdb.set_trace()
        
        sample_model = lambda _model: to_dense_adj(_model()).squeeze()
        baseline_model = Baseline(test_dataset)
        baseline_samples = [sample_model(baseline_model) for _ in range(N_samples_novelty)]
        

    elif args.mode == 'sample':
        model = build_model(node_feature_dim, args.embedding_dim, args.n_rounds, args.latent_dim, max_num_nodes)
        model.load_state_dict(torch.load('VAE_weights.pt', map_location=args.device))
        samples = model.sample(64)
        
        stats = GraphStatistics(samples[0])
        
        print(f'Degree: {stats.degree}')
        print(f'Eigenvector centrality: {stats.eigenvector_centrality}')
        print(f'Cluster coefficient: {stats.clustercoefficient}')
        
        

        
