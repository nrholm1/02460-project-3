from src.GraphVAE.vae_encoder_network import GNNEncoderNetwork, GRUGNNEncoderNetwork
from src.GraphVAE.vae_decoder_network import DecoderNetwork
from src.GraphVAE.basevae import VAE, GaussianEncoder, BernoulliDecoder, GaussianPrior
from src.graphgan import NDist
from src.GraphVAE.naivevae import NaiveVAE

def get_encoder_networks(node_feature_dim, embedding_dim, n_message_passing_rounds, M: int):
    """
    A function that creates encoder networks for a node-level VAE
    Keyword Arguments
    -----------------
        - node_feature_dim : Dimension of the node features
        - embedding_dim : Dimension of the node states
        - n_message_passing_rounds : Number of message passing rounds:
        - M: The dimension of the latent space.

    Returns:
        - mu_network and sigma_network.
    """
    # mu_network = GNNEncoderNetwork(node_feature_dim, embedding_dim, n_message_passing_rounds, M)
    # sigma_network = GNNEncoderNetwork(node_feature_dim, embedding_dim, n_message_passing_rounds, M)
    
    mu_network = GRUGNNEncoderNetwork(node_feature_dim, embedding_dim, n_message_passing_rounds, M)
    sigma_network = None

    return mu_network, sigma_network

def get_decoder_network(max_num_nodes: int, M: int):
    """
    A function that creates a decoder network for a node-level VAE
    """
    return DecoderNetwork(max_num_nodes, M)

def build_model(node_feature_dim, embedding_dim, 
                n_message_passing_rounds, M: int, 
                NDist_dataset, naive=False
                ):
    
    ndist = NDist(NDist_dataset)
    prior = GaussianPrior(M)
    
    mu_net, sigma_net = get_encoder_networks(node_feature_dim, embedding_dim, n_message_passing_rounds, M)
    encoder = GaussianEncoder(mu_net, sigma_net)
    
    decoder_net = get_decoder_network(ndist.max_nodes, M)
    decoder = BernoulliDecoder(decoder_net)

    if not naive:
        model = VAE(prior, decoder, encoder, ndist)
    else:
        model = NaiveVAE(prior, decoder, encoder, ndist)
    
    return model

    