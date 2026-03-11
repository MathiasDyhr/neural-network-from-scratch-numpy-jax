#%%########### PyNet_M10_Sweep: MNIST Hyperparameter Sweep ##############
"""
MNIST hyperparameter sweep script for WandB.
Works with PyNet_Sweep_Config.py to automatically test different configurations.

Usage:
    1. Run: wandb sweep PyNet_Sweep_Config.py
    2. Copy the sweep ID
    3. Run: wandb agent <sweep_id>
    
This script will be called automatically by WandB for each sweep run.
"""

import numpy as np
import wandb
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PyNet import PyNetBase, train, evaluate_model, plot_training_results, plot_confusion_matrix
np.random.seed(42)


def train_sweep():
    """
    Main training function called by WandB sweep agent.
    Gets hyperparameters from wandb.config and runs one training experiment.
    """
    
    # Initialize WandB run (sweep agent handles this automatically)
    run = wandb.init()
    
    # Get hyperparameters from sweep config
    config = wandb.config
    
    print("=" * 70)
    print(f"Starting sweep run: {run.name}")
    print("=" * 70)
    
    
    #%%######################### 1. Dataset Configuration ####################
    
    num_features = 28 * 28     # MNIST: 28x28 pixels
    num_classes = 10           # MNIST: digits 0-9
    
    
    #%%######################### 2. Load MNIST Data ##########################
    
    print("\nLoading MNIST dataset...")
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    
    # Split training into train/validation (90/10)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )
    
    # Reshape and normalize inputs
    X_train = X_train.reshape(-1, 28*28) / 255.0
    X_val = X_val.reshape(-1, 28*28) / 255.0
    X_test = X_test.reshape(-1, 28*28) / 255.0
    
    # One-hot encode labels
    T_train = to_categorical(y_train, num_classes=10)
    T_val = to_categorical(y_val, num_classes=10)
    T_test = to_categorical(y_test, num_classes=10)
    
    print(f"Training: {X_train.shape[0]:,} | Validation: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    
    
    #%%################### 3. Extract Sweep Hyperparameters ##################
    
    # Architecture
    hidden_units = config.hidden_units
    activation = config.activation
    weights_init = config.weights_init
    
    # Optimizer
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    
    # Regularization
    dropout_p_value = config.dropout_p_value
    # Create dropout list matching number of hidden layers
    num_hidden_layers = len(hidden_units)
    dropout_p = [dropout_p_value] * num_hidden_layers
    l2_coeff = config.l2_coeff
    
    # Training
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    loss = config.loss
    use_grad_clipping = config.use_grad_clipping
    max_grad_norm = config.max_grad_norm
    
    
    #%%################ 4. Print Configuration ####################
    
    print(f"\nConfiguration for this run:")
    print(f"  Architecture: {hidden_units} | {activation} | {weights_init}")
    print(f"  Optimizer: {optimizer} | LR: {learning_rate}")
    print(f"  Regularization: dropout={dropout_p_value} | L2={l2_coeff}")
    print(f"  Training: batch={batch_size} | epochs={num_epochs}")
    print()
    
    
    #%%################### 5. Initialize Neural Network ######################
    
    # Create MNIST-specific network class
    class PyNet_M10(PyNetBase):
        """MNIST-specific neural network using shared base functionality"""
        pass
    
    # Initialize network
    net = PyNet_M10(
        num_features, hidden_units, num_classes,
        weights_init, activation, loss, optimizer, l2_coeff, dropout_p,
        seed=getattr(config, 'seed', 42)  # Use config.seed if present, else default to 42
    )
    
    
    #%%########################### 6. Training Loop ##########################
    
    # Train the model (WandB is already initialized by sweep agent)
    # Pass None for wandb_project to prevent double initialization
    net.W, losses, train_accuracies, val_accuracies, val_losses = train(
        net, X_train.T, T_train.T, net.W,
        num_epochs, learning_rate, batch_size,
        X_val=X_val.T, T_val=T_val.T,
        use_clipping=use_grad_clipping,
        max_grad_norm=max_grad_norm,
        use_wandb=True,  # Always True for sweeps
        wandb_project=None,  # None = don't reinitialize, sweep agent already did it
        wandb_config=None,  # None = don't reinitialize, sweep agent already did it
        wandb_mode="online"  # Sweeps must be online
    )
    



    #%%########################## 7. Evaluate Model ##########################
    
    # Evaluate and display results
    y_pred, test_accuracy, test_loss = evaluate_model(
        net, X_test, T_test, y_test, net.W, train_accuracies, use_wandb=True
    )
    
    print(f"\n{'='*70}")
    print(f"Sweep run {run.name} complete!")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"{'='*70}\n")
    
    # Note: WandB run is finished by evaluate_model()


#%%######################### Main Execution ##################################

if __name__ == "__main__":
    # This function will be called by WandB sweep agent
    train_sweep()
