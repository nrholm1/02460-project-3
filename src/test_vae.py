
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch.utils.data import random_split

from src.utils import get_mutag_dataset, plot_adj #, drawnow
from src.nodevae import GaussianPrior, GaussianEncoder, BernoulliDecoder, GNNEncoder, GNNDecoder, VGAE

plt.ion() # Enable interactive plotting
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

def train(epochs, model, optimizer, criterion, scheduler, early_stopping_patience, train_loader, validation_loader, trial=None):
    
    train_accuracies, train_losses, validation_accuracies, validation_losses = [], [], [], []
    steps_since_last_loss_decrease = 0
    min_val_loss = float('inf')

    for epoch in range(epochs):
        # Loop over training batches
        model.train()
        train_accuracy = 0.
        train_loss = 0.
        for data in train_loader:
            loss = model(data.x, data.edge_index, batch=data.batch)
            #loss = criterion(out, data.y.float())

            # loss = 0
            # # Assuming data.y contains labels for each graph
            # for i, single_output in enumerate(out):
            #     target = data.y[i].expand_as(single_output)  # Ensure target is the same shape as output
            #     loss += criterion(single_output, target.float())  # Compute loss for each graph

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training loss and accuracy
            train_accuracy += sum((out>0) == data.y).detach().cpu() / len(train_loader.dataset)
            train_loss += loss.detach().cpu().item() * data.batch_size / len(train_loader.dataset)
        
        # Learning rate scheduler step
        scheduler.step()

        # Validation, print and plots
        with torch.no_grad():    
            model.eval()
            # Compute validation loss and accuracy
            validation_loss = 0.
            validation_accuracy = 0.
            for data in validation_loader:
                out = model(data.x, data.edge_index, data.batch)
                validation_accuracy += sum((out>0) == data.y).cpu() / len(validation_loader.dataset)
                validation_loss += criterion(out, data.y.float()).cpu().item() * data.batch_size / len(validation_loader.dataset)

            # Store the training and validation accuracy and loss for plotting
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)

            # Print stats and update plots
            if (epoch+1)%10 == 0:
                print(f'Epoch {epoch+1}')
                print(f'- Learning rate   = {scheduler.get_last_lr()[0]:.1e}')
                print(f'- Train. accuracy = {train_accuracy:.3f}')
                print(f'         loss     = {train_loss:.3f}')
                print(f'- Valid. accuracy = {validation_accuracy:.3f}')
                print(f'         loss     = {validation_loss:.3f}')

                plt.figure('Loss').clf()
                plt.plot(train_losses, label='Train')
                plt.plot(validation_losses, label='Validation')
                plt.legend()
                # show the hyperparams: (state_dim, num_message_passing_rounds, lr, gamma) with 3 actual decimals
                plt.title(f"Loss: state_d: {model.state_dim} | m_rounds: {model.num_message_passing_rounds} | lr: {optimizer.param_groups[0]['lr']:.4g} | gamma: {scheduler.gamma:.3g}")
                plt.xlabel('Epoch')
                plt.ylabel('Cross entropy')
                plt.yscale('log')
                #plt.yscale('linear')
                plt.tight_layout()
                drawnow()

                plt.figure('Accuracy').clf()
                plt.plot(train_accuracies, label='Train')
                plt.plot(validation_accuracies, label='Validation')
                plt.legend()
                plt.title(f"Accuracy: state_d: {model.state_dim} | m_rounds: {model.num_message_passing_rounds} | lr: {optimizer.param_groups[0]['lr']:.4g} | gamma: {scheduler.gamma:.3g}")
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.tight_layout()
                drawnow()
            
            # Early stopping check
            if validation_loss < min_val_loss:
                min_val_loss = validation_loss
                steps_since_last_loss_decrease = 0
            else:
                steps_since_last_loss_decrease += 1

            if steps_since_last_loss_decrease >= early_stopping_patience:
                print(f"Stopping early at epoch {epoch} due to no improvement in validation loss.")
                break
    
    return min_val_loss


if __name__ == '__main__':
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    torch.manual_seed(42)
    
    # Load data
    dataset = get_mutag_dataset(device)
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # dataloader for sampling single graphs
    
    # Split into training and validation
    rng = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)
    # Create dataloader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=100)
    validation_loader = DataLoader(validation_dataset, batch_size=44)
    test_loader = DataLoader(test_dataset, batch_size=44)
    
    # num_nodes = 1809
    # num_edges = 16744

    # Set hyperparameters
    node_feature_dim = 7 # number of features per node
    state_dim = 16 #dimension of state space 
    num_message_passing_rounds = 4 #number of message passing rounds

    latent_dim = 8 # dimension of latent space
    GRU = False

    early_stopping_patience = 10 # epochs to wait for validation loss to decrease before stopping
    epochs = 200

    # Instantiate model
    prior = GaussianPrior(latent_dim)
    
    mu_encoder_net = GNNEncoder(node_feature_dim, state_dim, num_message_passing_rounds, GRU, latent_dim)
    std_encoder_net = GNNEncoder(node_feature_dim, state_dim, num_message_passing_rounds, GRU, latent_dim)
    encoder = GaussianEncoder(mu_encoder_net, std_encoder_net)

    decoder_net = GNNDecoder(latent_dim)
    decoder = BernoulliDecoder(decoder_net)

    nodeVGAE_model = VGAE(prior=prior, encoder=encoder, decoder=decoder, k=1)

    criterion = torch.nn.BCEWithLogitsLoss() # computes sigmoid and then BCELoss   
    #criterion = torch.nn.BCELoss() # 
    optimizer = torch.optim.Adam(nodeVGAE_model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    #rows order - either  
        # avg degree or 
        # cross entropy

    # Train model
    min_val_loss = train(epochs, nodeVGAE_model, optimizer, criterion, scheduler, early_stopping_patience, train_loader, validation_loader)
    print(f"Minimum validation loss: {min_val_loss}")

    # Save model
    with torch.no_grad():
        data = next(iter(test_loader))
        out = nodeVGAE_model(data.x, data.edge_index, data.batch).cpu()
        torch.save(out, 'test_predictions.pt')


