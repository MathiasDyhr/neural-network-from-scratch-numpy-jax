#%%########### 1. Import Required Libraries and Configuration ############## 

import warnings
import numpy as np
import tensorflow_datasets as tfds
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PyNet import PyNetBase, train, evaluate_model, plot_training_results, plot_confusion_matrix
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress runtime warnings for mental stability

# Dataset Configuration
num_features = 28 * 28     # EMNIST: 28x28 pixels
num_classes = 47           # EMNIST Balanced: 47 classes (merged digits and letters)

# Architecture Configuration
hidden_units = [32, 32]    # Units per hidden layer [layer1, layer2, ...]
activation = 'relu'        # Activation function: 'relu', 'tanh', 'sigmoid'
weights_init = 'he'        # Weight initialization: 'he', 'xavier', 'normal'

# Training Configuration  
num_epochs = 10           # Number of training epochs
learning_rate = 0.001      # Learning rate for gradient descent
batch_size = 32            # Mini-batch size
loss = 'cross_entropy'     # Loss function: 'cross_entropy', 'mse', 'mae'
optimizer = 'adam'         # Optimizer: 'sgd', 'adam', 'rmsprop'
l2_coeff = 1e-8            # L2 regularization coefficient (weight_decay)
dropout_p = [0.1, 0.1]     # Dropout probabilities per layer [hidden1, hidden2, ...]; 0.0 = no dropout
use_grad_clipping = False  # Enable/disable gradient clipping
max_grad_norm = 50.0       # Maximum gradient norm for clipping

# WandB Configuration
use_wandb = False                           # Enable W&B logging
wandb_project = "02456-project"             # Your W&B project name
wandb_mode = "offline"                      # W&B mode: "online", "offline", or "disabled"
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
    "dataset": "EMNIST",
    "framework": "PyNet"
}

#%%######################## 2. Load EMNIST Data ############################

# Load EMNIST Balanced dataset using TensorFlow Datasets
print("Loading EMNIST Balanced dataset...")
ds_train, ds_test = tfds.load('emnist/balanced', split=['train', 'test'], as_supervised=True)

# Convert to numpy arrays
def preprocess_data(ds):
    images, labels = [], []
    for image, label in ds:
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

print("Converting to numpy arrays...")
X_train_full, y_train_full = preprocess_data(ds_train)
X_test, y_test = preprocess_data(ds_test)

# Split training into train/validation (90/10)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
)

# Reshape and normalize inputs (same as MNIST)
X_train = X_train.reshape(-1, 28*28) / 255.0
X_val = X_val.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# EMNIST balanced uses labels 0-46 (47 classes)
print(f"Label range: {y_train.min()}-{y_train.max()}")

# One-hot encode labels (0-46 for 47 classes)
T_train = to_categorical(y_train, num_classes=47)
T_val = to_categorical(y_val, num_classes=47)
T_test = to_categorical(y_test, num_classes=47)

print(f"Successfully loaded!")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Validation samples: {X_val.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")
print(f"Classes: 0-9 + merged letters (47 total)")
print(f"Image shape: 28x28 → {X_train.shape[1]} features")




#%%################ 3. Initialize EMNIST Neural Network #####################

# Create EMNIST Balanced-specific network class
class PyNet_E47B(PyNetBase):
    """EMNIST Balanced-specific neural network using shared base functionality"""
    pass

# Initialize network
net = PyNet_E47B(num_features, hidden_units, num_classes, weights_init, activation, loss, optimizer, l2_coeff, dropout_p)

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

# Convert some predictions to letters for demonstration  
def number_to_letter(num):
    return chr(ord('A') + num)

print(f"\n Sample Letter Predictions:")
sample_indices = np.random.choice(len(y_test), 5, replace=False)
for i in sample_indices:
    true_letter = number_to_letter(y_test[i])  # y_test is already 0-25 range
    pred_letter = number_to_letter(y_pred[i])
    print(f"True: {true_letter}, Predicted: {pred_letter}")




#%%######################## 6. Plot Training Results #######################

# Plot training curves
plot_training_results(
    losses=losses,
    train_accuracies=train_accuracies,
    val_accuracies=val_accuracies,
    val_losses=val_losses,
    test_accuracy=test_accuracy,
    figsize=(15, 5),
    save_path=None  # Set to a path like 'emnist_balanced_training.png' to save
)




#%%###################### 7. Plot Confusion Matrix ########################

# EMNIST Balanced has 47 classes with merged similar characters
# Actual mapping:
# Classes 0-9: digits 0-9
# Classes 10-35: uppercase letters A-Z
# Classes 36-46: lowercase letters that look different from uppercase: a,b,d,e,f,g,h,n,q,r,t

# Lowercase letters kept in EMNIST Balanced (look different from uppercase)
lowercase_letters = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

class_names = []
for i in range(47):
    if i < 10:
        # Digits 0-9
        class_names.append(str(i))
    elif i < 36:
        # Uppercase letters A-Z (indices 10-35)
        class_names.append(chr(ord('A') + (i - 10)))
    else:
        # Lowercase letters (indices 36-46)
        class_names.append(lowercase_letters[i - 36])

print(f"\nConfusion Matrix Class Names:")
print(f"  0-9: Digits")
print(f"  A-Z: Uppercase letters (26 classes)")
print(f"  a,b,d,e,f,g,h,n,q,r,t: Lowercase letters (11 classes)")
print(f"  Note: Other lowercase letters merged with uppercase due to similarity")

# Plot confusion matrix
plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    class_names=class_names,  # 0-9, A-Z, a-k
    normalize=False,
    figsize=(12, 10),
    save_path=None  # Set to a path like 'emnist_balanced_confusion.png' to save
)