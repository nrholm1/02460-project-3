import torch
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj
import pdb

def get_inputs(batch, edge_index, max_num_nodes = None):
    ### Get Adjacency matrices and node masks
    num_nodes_per_graph = torch.unique(batch, return_counts=True)[1]
    max_num_nodes = num_nodes_per_graph.max().item() if max_num_nodes is None else max_num_nodes

    Adj = to_dense_adj(edge_index, batch, max_num_nodes=max_num_nodes, batch_size=batch.max().item()+1)
    node_masks = torch.zeros_like(Adj)
    
    for i, m in enumerate(num_nodes_per_graph):
        node_masks[i, :m, :m] = 1
    
    node_masks = node_masks.to(torch.long)

    return Adj, node_masks

class VAETrainer:
    def __init__(self, model, optimizer, data_loader, epochs, device) -> None:
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.epochs = epochs
        self.device = device
        self.model.to(self.device)
    
    def train(self):
        """
        Train a VAE model.

        Parameters:
        model: [VAE]
        The VAE model to train.
        optimizer: [torch.optim.Optimizer]
            The optimizer to use for training.
        data_loader: [torch.utils.data.DataLoader]
                The data loader to use for training.
        epochs: [int]
            Number of epochs to train for.
        device: [torch.device]
            The device to use for training.
        """

        self.model.train()
        num_steps = len(self.data_loader) * self.epochs
        epoch = 0
        losses = []

        _x = next(iter(self.data_loader))
        Adj, node_masks = get_inputs(_x.batch, _x.edge_index)
        
        with tqdm(range(num_steps)) as pbar:
            for step in pbar:
                data = next(iter(self.data_loader))
                data = data.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model(data.x, data.edge_index, data.batch, Adj, node_masks)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

                # Report
                if step % 5 ==0 :
                    loss = loss.detach().cpu()
                    pbar.set_description(f"epoch={epoch}, step={step}, loss={torch.mean(torch.tensor(losses)):.1f}")

                if (step+1) % len(self.data_loader) == 0:
                    epoch += 1
        torch.save(self.model.state_dict(), 'VAE_weights.pt')

        