# TODO

- [X] Baseline Model - Erdös-Rényi
    - [X] Compute empirical distribution of number of nodes
    - [X] Compute empirical distribution of link probability given number of nodes
    - [X] Sample number of nodes from empirical distribution
    - [X] Compute link probability $r = |E| / N^2$ given the graphs in the training set with N nodes.
    - [X] Sample graphs according Generate edges according to the generative model [1. sample N, 2. generate edges]

- [ ] Deep Generative Model(s) - one or more, basically
    - [ ] VAEs
        - [ ] VAE - node level latents
            - [ ] Message passing NN
            - [ ] Graph Convolutional NN
        - [ ] VAE - graph level latents
            - [ ] Message passing NN
            - [ ] Graph Convolutional NN
    - [ ] GAN
        - [ ] Message passing NN
        - [ ] Graph Convolutional NN

- [ ] Evaluation metrics
    - [X] Uniqueness
    - [X] Novelty
    - [ ] Uniqueness and Novelty (at the same time)
- [ ] Simple graph statistics
    - [X] Node degree
    - [X] Clustering coefficient
    - [X] Eigenvector centrality
    - [ ] Plot histograms of these
