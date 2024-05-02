import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import networkx as nx
import numpy as np
import pdb

from src.graph_statistics import GraphStatistics
from src.graphgan import create_gan_model, NDist
from graph_statistics_nx_sanity import convert_geometric_to_network_x
from src.baseline import Baseline
from src.utils import get_mutag_dataset, plot_adj
from src.GraphVAE.graphvae import build_model



def adjacency_to_edge_index(adjacency_matrix):
    # Convert the adjacency matrix to a tensor if it is not already
    adj = torch.tensor(adjacency_matrix)
    
    # Find the indices of non-zero entries
    src, dst = adj.nonzero(as_tuple=True)
    
    # Stack them into a 2xm tensor
    edge_index = torch.stack([src, dst], dim=0)
    
    return edge_index

class SampleGenerator:
    def __init__(self, dataset):
        self.dataset = dataset

    def __sample_baseline(self, model: Baseline):
        sample_model = lambda _model: to_dense_adj(_model()).squeeze()
        return sample_model(model)

    def make_baseline_samples(self, model: Baseline, num_samples: int, save_folder: str = "samples/baseline/"):
        # full dataset dataloader
        # make n samples and save make a dataset of them
        samples = []
        for i in tqdm(range(num_samples)):
            dense_adj_sample = self.__sample_baseline(model)
            samples.append(dense_adj_sample)
            save_path = os.path.join(save_folder, f"sample_{i}.pt")
            self.save_sample(dense_adj_sample, save_path)

        return samples
        
    def make_gan_samples(self, model, num_samples: int, save_folder: str = "samples/gan/"):
        samples = []
        gen = model.generator

        with torch.no_grad():
            for i in tqdm(range(num_samples)):
                # sample from the GAN model
                sample = gen.sample(1)
                # convert to dense adjacency matrix
                sample = to_dense_adj(sample.edge_index).squeeze()
                samples.append(sample)
                save_path = os.path.join(save_folder, f"sample_{i}.pt")
                self.save_sample(sample, save_path)

        return samples
    
    def make_vae_samples(self, vae_model, num_samples: int, save_folder: str = 'samples/vae'):
        print("VAE!! ", save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        samples = vae_model.sample(num_samples)
        for i, Adj in enumerate(samples):
            self.save_sample(Adj, f"{save_folder}/sample_{i}.pt")
        return samples

    def save_sample(self, adj, path):
        # create folder if it does not exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(adj, path)

def make_histograms(dataset, sample_folders, plot_colors, save_path="samples/histograms.pdf"):

    # make histogram plots
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    for label, sample_folder in sample_folders.items():
        path = sample_folder

        if label.lower() == "dataset":
            samples =  [to_dense_adj(data.edge_index).squeeze() for data in dataset]
        else:
            samples = [torch.load(os.path.join(path, f)) for f in os.listdir(path)]

        # Make Graph Statistics
        graph_statistics = [GraphStatistics(samples) for samples in samples]

        avg_degrees = [graph_stat.avg_degree for graph_stat in graph_statistics]
        avg_clustercoefficients = [graph_stat.avg_clustercoefficient for graph_stat in graph_statistics]
        avg_eigenvector_centralities = [graph_stat.avg_eigenvector_centrality for graph_stat in graph_statistics]

        # Calculate weights to display the histogram as percentages
        weights_degrees = np.ones_like(avg_degrees) / len(avg_degrees)
        weights_clustercoefficients = np.ones_like(avg_clustercoefficients) / len(avg_clustercoefficients)
        weights_eigenvector_centralities = np.ones_like(avg_eigenvector_centralities) / len(avg_eigenvector_centralities)

        # Define bin width
        

        # Determine the global minimum and maximum across all datasets for each statistic
        bin_width_degrees = 0.1
        min_degree, max_degree = min(avg_degrees), max(avg_degrees)
        bins_degrees = np.arange(min_degree, max_degree + bin_width_degrees, bin_width_degrees)

        bin_widht_clustercoefficients = 0.02
        min_cluster, max_cluster = min(avg_clustercoefficients), max(avg_clustercoefficients)
        bins_clustercoefficients = np.arange(min_cluster, max_cluster + bin_widht_clustercoefficients, bin_widht_clustercoefficients)

        bin_width_eigenvector_centralities = 0.01
        min_eigenvector, max_eigenvector = min(avg_eigenvector_centralities), max(avg_eigenvector_centralities)
        bins_eigenvector_centralities = np.arange(min_eigenvector, max_eigenvector + bin_width_eigenvector_centralities, bin_width_eigenvector_centralities)

        # Average Degree Histogram
        axs[0].hist(avg_degrees, bins=bins_degrees, weights=weights_degrees, color=plot_colors[label], alpha=0.3, edgecolor='black', linewidth=1.2, label=label)
        # Average Cluster Coefficient Histogram
        axs[1].hist(avg_clustercoefficients, bins=bins_clustercoefficients, weights=weights_clustercoefficients, color=plot_colors[label], alpha=0.3, edgecolor='black', linewidth=1.2, label=label)        
        # Average Eigenvector Centrality Histogram
        axs[2].hist(avg_eigenvector_centralities, bins=bins_eigenvector_centralities, weights=weights_eigenvector_centralities, color=plot_colors[label], alpha=0.3, edgecolor='black', linewidth=1.2, label=label)

    axs[0].set_title("Average Degree Histogram")
    axs[0].set_xlabel("Degree")
    axs[0].set_ylabel("Percentage (%)")

    axs[1].set_title("Average Cluster Coefficient Histogram")
    axs[1].set_xlabel("Cluster Coefficient")
    axs[1].set_ylabel("Percentage (%)")
    axs[1].set_xlim(left=0)

    axs[2].set_title("Average Eigenvector Centrality Histogram")
    axs[2].set_xlabel("Eigenvector Centrality")
    axs[2].set_ylabel("Percentage (%)")

    plt.tight_layout()
    plt.legend()

    plt.savefig(save_path)


def make_histograms_rows(dataset, sample_folders, plot_colors, save_path="samples/histograms.pdf"):
    # make histogram of four rows and three columns
    n_rows = len(sample_folders)
    fig, axs = plt.subplots(n_rows, 3, figsize=(6, 5))


    global_min_degree = float('inf')
    global_max_degree = float('-inf')
    global_min_cluster = float('inf')
    global_max_cluster = float('-inf')
    global_min_eigenvector = float('inf')
    global_max_eigenvector = float('-inf')

    max_y_degrees = 0
    max_y_clustercoefficients = 0
    max_y_eigenvector_centralities = 0


    for n, (label, sample_folder) in enumerate(sample_folders.items()):
        path = sample_folder

        if label.lower() == "dataset":
            samples =  [to_dense_adj(data.edge_index).squeeze() for data in dataset]
        else:
            samples = [torch.load(os.path.join(path, f)) for f in os.listdir(path)]

        # Make Graph Statistics
        graph_statistics = [GraphStatistics(samples) for samples in samples]

        avg_degrees = [graph_stat.avg_degree for graph_stat in graph_statistics]
        avg_clustercoefficients = [graph_stat.avg_clustercoefficient for graph_stat in graph_statistics]
        avg_eigenvector_centralities = [graph_stat.avg_eigenvector_centrality for graph_stat in graph_statistics]

        # Calculate weights to display the histogram as percentages
        weights_degrees = np.ones_like(avg_degrees) / len(avg_degrees)
        weights_clustercoefficients = np.ones_like(avg_clustercoefficients) / len(avg_clustercoefficients)
        weights_eigenvector_centralities = np.ones_like(avg_eigenvector_centralities) / len(avg_eigenvector_centralities)

        # Determine the global minimum and maximum across all datasets for each statistic
        bin_width_degrees = 0.5
        min_degree, max_degree = min(avg_degrees), max(avg_degrees)
        global_min_degree, global_max_degree = min(global_min_degree, min_degree), max(global_max_degree, max_degree)
        bins_degrees = np.arange(min_degree, max_degree + bin_width_degrees, bin_width_degrees)

        bin_widht_clustercoefficients = 0.04
        min_cluster, max_cluster = min(avg_clustercoefficients), max(avg_clustercoefficients)
        global_min_cluster, global_max_cluster = min(global_min_cluster, min_cluster), max(global_max_cluster, max_cluster)
        bins_clustercoefficients = np.arange(min_cluster, max_cluster + bin_widht_clustercoefficients, bin_widht_clustercoefficients)
        if bins_clustercoefficients.shape[0] == 1 and bins_clustercoefficients[0] == 0:
            # make the widtht of bin_widht_clustercoefficients around 0
            bins_clustercoefficients = np.array([-bin_widht_clustercoefficients/2, bin_widht_clustercoefficients/2])

        bin_width_eigenvector_centralities = 0.02
        min_eigenvector, max_eigenvector = min(avg_eigenvector_centralities), max(avg_eigenvector_centralities)
        global_min_eigenvector, global_max_eigenvector = min(global_min_eigenvector, min_eigenvector), max(global_max_eigenvector, max_eigenvector)
        bins_eigenvector_centralities = np.arange(min_eigenvector, max_eigenvector + bin_width_eigenvector_centralities, bin_width_eigenvector_centralities)
        # Average Degree Histogram
        axs[n,0].hist(avg_degrees, bins=bins_degrees,color=plot_colors[label], alpha=0.3, edgecolor='black', linewidth=1.2, label=label)
        # Average Cluster Coefficient Histogram
        axs[n,1].hist(avg_clustercoefficients, bins=bins_clustercoefficients, color=plot_colors[label], alpha=0.3, edgecolor='black', linewidth=1.2, label=label)        
        # Average Eigenvector Centrality Histogram
        axs[n,2].hist(avg_eigenvector_centralities, bins=bins_eigenvector_centralities, color=plot_colors[label], alpha=0.3, edgecolor='black', linewidth=1.2, label=label)

        # get max y value for each plot
        max_y_degrees = max(max_y_degrees, axs[n, 0].get_ylim()[1])
        max_y_clustercoefficients = max(max_y_clustercoefficients, axs[n, 1].get_ylim()[1])
        max_y_eigenvector_centralities = max(max_y_eigenvector_centralities, axs[n, 2].get_ylim()[1])
        
        if n == 0:
            axs[n, 0].set_title("Avg. Degree Hist.", fontsize=8)
        if n == len(sample_folders) - 1:
            axs[n, 0].set_xlabel("Degree", fontsize=6)
        # axs[n, 0].set_ylabel("Frequency")
        # axs[n, 0].set_ylabel(r'\\textbf{' + label + '}' + '\n' + r'\textnormal{Percentage (\%)}', fontsize=12)

            # Set the y-axis label with normal font weight
        axs[n, 0].set_ylabel("Frequency", fontsize=6)

        axs[n, 0].text(-0.5, 0.5, label, transform=axs[n, 0].transAxes, fontsize=10, fontweight='bold', va='center', ha='right', rotation=90)


        if n == 0:
            axs[n, 1].set_title("Avg. Cluster Coefficient Hist.", fontsize=8)
        if n == len(sample_folders) - 1:
            axs[n, 1].set_xlabel("Cluster Coefficient", fontsize=6)
        # axs[n, 1].set_ylabel("Percentage (%)")
        axs[n, 1].set_xlim(left=0)

        if n == 0:
            axs[n, 2].set_title("Avg. Eigenv. Centrality Hist.", fontsize=8)
        if n == len(sample_folders) - 1:
            axs[n, 2].set_xlabel("Eigenvector Centrality", fontsize=6)
    
        # set a rotated title for the y-axis

    for i in range(len(sample_folders)):
        axs[i, 0].set_xlim(global_min_degree, global_max_degree)
        axs[i, 1].set_xlim(global_min_cluster, global_max_cluster)
        axs[i, 2].set_xlim(global_min_eigenvector, global_max_eigenvector)

        if i != 0:
            axs[i, 0].set_ylim(0, max_y_degrees)
            axs[i, 1].set_ylim(0, max_y_clustercoefficients)
            axs[i, 2].set_ylim(0, max_y_eigenvector_centralities)

    plt.tight_layout()  
    plt.savefig(save_path)


if __name__ == "__main__":
    gan_model_path = "models/ep1500_GraphGAN.pt"
    vae_model_path = "models/VAE_weights.pt"
    sample_folders = {"dataset": None, 
                      "baseline": "samples/baseline/", 
                      "gan": "samples/gan/",
                      "vae": "samples/vae"}
    plot_colors = {"dataset": "lightgreen", "baseline": "skyblue", "gan": "lightcoral", "vae": "yellow"}


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--histogram", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--plot-grid", default=None, type=str, choices=[None, "baseline", "gan", "vae"], help="Choose which model to plot the grid from.")
    parser.add_argument("--plot-mean", default=None, type=str, choices=[None, "baseline", "gan", "vae"], help="Choose which model to plot the mean graph from.")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--gan-model-path", type=str, default=None)
    parser.add_argument("--gan-statedim", type=int, default=10)
    parser.add_argument("--gan-mp-rounds", type=int, default=5)
    parser.add_argument("--vae-model-path", type=str, default=None)
    parser.add_argument("--vae-embedding-dim", type=int, default=16)
    parser.add_argument("--vae-M", type=int, default=16)
    parser.add_argument("--vae-n-message-passing-rounds", type=int, default=25)

    
    args = parser.parse_args()

    # print args nicely
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    dataset = get_mutag_dataset()
    sample_generator = SampleGenerator(dataset)
    baseline_model = Baseline(sample_generator.dataset)

    if args.gan_model_path:
        gan_model = create_gan_model(dataset, mp_rounds=args.gan_mp_rounds, state_dim=args.gan_statedim)
        state_dict = torch.load(args.gan_model_path)
        gan_model.load_state_dict(state_dict)

    if args.vae_model_path:
        ndist = NDist(dataset)
        vae_model = build_model(node_feature_dim=7, embedding_dim=args.vae_embedding_dim, M=args.vae_M,
                                n_message_passing_rounds=args.vae_n_message_passing_rounds, NDist_dataset=dataset)
        vae_model.load_state_dict(torch.load(args.vae_model_path, map_location="cpu"))

    if args.sample:
        sample_generator.make_baseline_samples(baseline_model, args.num_samples, sample_folders["baseline"])
        if args.gan_model_path:
            sample_generator.make_gan_samples(gan_model, args.num_samples, sample_folders["gan"])
        if args.vae_model_path:
            sample_generator.make_vae_samples(vae_model, args.num_samples, sample_folders["vae"])
   
    # remove vae from the sample_folders if the model is not loaded
    if not args.vae_model_path:
        sample_folders.pop("vae")
        plot_colors.pop("vae")
    # remove gan from the sample_folders if the model is not loaded
    if not args.gan_model_path:
        sample_folders.pop("gan")
        plot_colors.pop("gan")

    # make_histograms(dataset, sample_folders, plot_colors)
    if args.histogram:
        make_histograms_rows(dataset, sample_folders, plot_colors, save_path="samples/histograms_rows.pdf")

    if args.table:
        from src.eval_metrics import compute_graph_hashes, eval_novelty, eval_unique, eval_novel_and_unique

        dataset_hashes = compute_graph_hashes([to_dense_adj(data.edge_index).squeeze() for data in dataset], return_list=True)
        # make a list of the dataset and add half of the dataset to the list

        for label, sample_folder in sample_folders.items():
            if label.lower() != "dataset":
                samples = [torch.load(os.path.join(sample_folder, f)) for f in os.listdir(sample_folder)]

                gen_graph_hashes = compute_graph_hashes(samples, return_list=True)
                novelty = eval_novelty(gen_graph_hashes, dataset_hashes)
                uniqueness = eval_unique(gen_graph_hashes)
                novelty_and_unique = eval_novel_and_unique(gen_graph_hashes, dataset_hashes)
                print(f"Novelty for {label}: {novelty}")
                print(f"Uniqueness for {label}: {uniqueness}")
                print(f"Novelty and Uniqueness for {label}: {novelty_and_unique}")

    if args.plot_grid:
        from src.utils import plot_adj
        import matplotlib.pyplot as plt

        sample_folder = sample_folders[args.plot_grid]
        samples = [torch.load(os.path.join(sample_folder, f)) for f in os.listdir(sample_folder)]

        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(4):
            for j in range(4):
                adj = samples[i*4 + j]

                plot_adj(adj, ax=axs[i, j])
        
        # remove axis labels
        for ax in axs.flat:
            ax.set(xticks=[], yticks=[])

        # remove titles
        for ax in axs.flat:
            ax.set_title("")

        plt.tight_layout()
        plt.show()

    if args.plot_mean:
        from src.utils import plot_adj
        import matplotlib.pyplot as plt

        sample_folder = sample_folders[args.plot_mean]
        samples = [torch.load(os.path.join(sample_folder, f)) for f in os.listdir(sample_folder)]

        max_size = max([sample.size(0) for sample in samples])

        mean_adj = torch.zeros(max_size, max_size)
        for sample in samples:
            mean_adj[:sample.size(0), :sample.size(0)] += sample
        mean_adj /= len(samples)

        plt.imshow(mean_adj, cmap='Greys', vmin=0, vmax=torch.max(mean_adj).item())
        plt.show()

    