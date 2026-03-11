#%%########### 1. Import Required Libraries and Configuration ##############    

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PyNet import PyNetBase, train, evaluate_model, plot_training_results, plot_confusion_matrix

# Dataset Configuration
num_features = 28 * 28     # MNIST: 28x28 pixels
num_classes = 10           # MNIST: digits 0-9

# Architecture Configuration
hidden_units = [128, 128]    # Units per hidden layer [layer1, layer2, ...]
activation = 'relu'        # Activation function: 'relu', 'tanh', 'sigmoid'
weights_init = 'he'        # Weight initialization: 'he', 'xavier', 'normal'

# Training Configuration  
num_epochs = 100           # Number of training epochs
learning_rate = 0.001      # Learning rate for gradient descent
batch_size = 32            # Mini-batch size
loss = 'cross_entropy'     # Loss function: 'cross_entropy', 'mse', 'mae'
optimizer = 'adam'         # Optimizer: 'sgd', 'adam', 'rmsprop'
l2_coeff = 1e-8            # L2 regularization coefficient (weight_decay)
dropout_p = [0.3, 0.3]     # Dropout probabilities per layer [hidden1, hidden2, ...]; 0.0 = no dropout
use_grad_clipping = False  # Enable/disable gradient clipping
max_grad_norm = 50.0       # Maximum gradient norm for clipping

# WandB Configuration
use_wandb = True                           # Enable W&B logging
wandb_project = "PyNet_M10_manual"             # Your W&B project name
wandb_mode = "online"                      # W&B mode: "online", "offline", or "disabled"
wandb_config = {
    # Architecture
    "num_features": num_features,
    "hidden_units": hidden_units,
    "num_classes": num_classes,
    "activation": activation,
    "weights_init": weights_init,

    # Training
    "optimizer": optimizer,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "loss": loss,
    "l2_coeff": l2_coeff,
    "dropout_p": dropout_p,
    "use_grad_clipping": use_grad_clipping,
    "max_grad_norm": max_grad_norm,

    # Metadata
    "dataset": "MNIST",
    "framework": "PyNet"
}




#%%######################### 2. Load MNIST Data ############################

# Load MNIST dataset
print("Loading MNIST dataset...")
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

print(f"Successfully loaded!")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Validation samples: {X_val.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")
print(f"Classes: 0-9 (10 total)")
print(f"Image shape: 28x28 → {X_train.shape[1]} features")




#%%################ 3. Initialize MNIST Neural Network #####################

# Create MNIST-specific network class
class PyNet_M10(PyNetBase):
    """MNIST-specific neural network using shared base functionality"""
    pass

# Initialize network
net = PyNet_M10(num_features, hidden_units, num_classes, weights_init, activation, loss, optimizer, l2_coeff, dropout_p, seed=42)

print(f"\nNetwork Architecture:")
print(f"   Input features: {num_features}")
print(f"   Hidden layers: {hidden_units}")
print(f"   Output classes: {num_classes}")
print(f"   Activation: {activation}")
print(f"   Weight init: {weights_init}")
print(f"Training Configuration:")
print(f"   Optimizer: {optimizer}")
print(f"   Learning rate: {learning_rate}")
print(f"   Batch size: {batch_size}")
print(f"   Epochs: {num_epochs}")
print(f"   Loss function: {loss}")
print(f"   L2 coefficient: {l2_coeff}")
print(f"   Dropout probabilities: {dropout_p}")
print(f"   Gradient clipping: {use_grad_clipping}")
print(f"   Max gradient norm: {max_grad_norm}")




#%%########################### 4. Training Loop ############################

# Train the model (using configured gradient clipping)
net.W, losses, train_accuracies, val_accuracies, val_losses = train(
    net, X_train.T, T_train.T, net.W,
    num_epochs, learning_rate, batch_size,
    X_val=X_val.T, T_val=T_val.T,
    use_clipping=use_grad_clipping, max_grad_norm=max_grad_norm,
    use_wandb=use_wandb,
    wandb_project=wandb_project,
    wandb_config=wandb_config,
    wandb_mode=wandb_mode
)

#%%########################## 5. Evaluate Model ############################

# Evaluate and display results
y_pred, test_accuracy, test_loss = evaluate_model(
    net, X_test, T_test, y_test, net.W, train_accuracies, use_wandb=use_wandb
)




#%%######################## 6. Plot Training Results #######################

# Plot training curves
plot_training_results(
    losses=losses,
    train_accuracies=train_accuracies,
    val_accuracies=val_accuracies,
    val_losses=val_losses,
    test_accuracy=test_accuracy,
    figsize=(15, 5),
    save_path=None  # Set to a path like 'mnist_training.png' to save
)




#%%###################### 7. Plot Confusion Matrix ########################

# Plot confusion matrix
plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    class_names=[str(i) for i in range(10)],  # MNIST digits 0-9
    normalize=False,
    figsize=(8, 6),
    save_path=None  # Set to a path like 'mnist_confusion.png' to save
)