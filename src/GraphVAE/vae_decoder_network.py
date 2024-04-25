import torch.nn as nn
import torch


class DecoderNetwork(nn.Module):
    def __init__(self, num_nodes: int, M: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.M = M

        self.net = torch.nn.Sequential(nn.Linear(self.M, self.M),
                                       nn.ReLU(),
                                       nn.Linear(self.M, self.num_nodes),
                                       nn.ReLU(),
                                       nn.Linear(self.num_nodes, self.num_nodes*self.num_nodes // 2),
                                       nn.ReLU(),
                                       nn.Linear(self.num_nodes*self.num_nodes // 2, self.num_nodes*self.num_nodes))

    def forward(self, Z):
        """
        Keyword Arguments
        -----------------
        Z: A matrix of latent features with dimension |V| x M
        
        Returns: A matrix of size batch_size x |V| x |V|
        """
        # out = Z@Z.T
        # if (out==torch.inf).any().item() or (out==torch.inf).any.item():
        #     return torch.zeros_like(out)
        X = self.net(Z)
        X = X.view(-1, self.num_nodes, self.num_nodes)
        
        return X