import torch
import pdb
from functools import cached_property

class GraphStatistics:
    def __init__(self, A: torch.tensor) -> None:
        self.A = A
        assert A.shape[0] == A.shape[1], 'The inputted adjacency matrix is not symmetric'
    
    @cached_property
    def degree(self):
        return torch.diag(self.A @ self.A)
    
    @cached_property
    def avg_degree(self):
        return torch.mean(self.degree)

    @cached_property
    def eigenvector_centrality(self):
        e_vals, e_vecs = torch.linalg.eigh(self.A)
        return abs(e_vecs[:, torch.argmax(e_vals)])
    
    @cached_property
    def avg_eigenvector_centrality(self):
        return torch.mean(self.eigenvector_centrality)

    @cached_property
    def clustercoefficient(self):
        I = torch.eye(self.A.size(1))
        D = self.degree * I
        A_cubed = self.A @ self.A @ self.A
        
        inverse_term = 1/torch.diag(D @ (D-I))
        inverse_term[inverse_term == torch.inf] = 0. # handles zero-division
        cc = (inverse_term * I) @ torch.diag(A_cubed)
        return cc
    
    @cached_property
    def avg_clustercoefficient(self):
        return torch.mean(self.clustercoefficient)
    
def graph_statistics_example():
    """
    Function that runs an MVP example of the GraphStatistics class.
    The adjacency matrix is the one from slide 27 of week 9 (example with cluster coefficient).
    """
    A_ex = torch.tensor([[0,1,0,0,0], 
                         [1,0,1,1,1],
                         [0,1,0,1,1],
                         [0,1,1,0,1],
                         [0,1,1,1,0]], dtype=torch.float32)
    GS = GraphStatistics(A_ex)
    print('\nMVP Graph statistics example:\n')
    print(f'Degree: {GS.degree}')
    print(f'Eigenvector centrality: {GS.eigenvector_centrality}')
    print(f'Cluster coefficient: {GS.clustercoefficient}\n')

def graph_statistics_example2():
    A = torch.tensor([[0,0,1,1,0,1,0],
                      [0,0,0,0,1,1,1],
                      [1,0,0,1,0,1,0],
                      [1,0,1,0,0,1,0],
                      [0,1,0,0,0,1,1],
                      [1,1,1,1,1,0,1],
                      [0,1,0,0,1,1,0]], dtype=torch.float32)
    GS = GraphStatistics(A)
    print('\nMVP Graph statistics example:\n')
    print(f'Degree: {GS.degree}')
    print(f'Eigenvector centrality: {GS.eigenvector_centrality}')
    print(f'Cluster coefficient: {GS.clustercoefficient}\n')
    

if __name__ == '__main__':
    graph_statistics_example2()