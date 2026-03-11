#%% JAXNet Shared Functions Module
import time
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from functools import partial
import wandb

class JAXNetBase:
    """Base class containing all shared neural network functionality"""
    
    def __init__(self, num_features, hidden_units, num_output, weights_init='he', activation='relu', loss='cross_entropy', optimizer='sgd', l2_coeff=0.0, dropout_p=None, seed=42):
        """
        Initialize neural network with configurable architecture.
        
        Args:
            num_features: Number of input features
            hidden_units: List of hidden layer sizes [layer1, layer2, ...]
            num_output: Number of output classes
            weights_init: Weight initialization method ('he', 'xavier', 'normal')
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            loss: Loss function ('cross_entropy', 'mse', 'mae')
            optimizer: Optimizer type ('sgd', 'adam', 'rmsprop')
            l2_coeff: L2 regularization coefficient (weight_decay)
            dropout_p: List of dropout probabilities for each hidden layer (None = no dropout)
            seed: Random seed for weight initialization
        """
        
        # Build layer sizes: input → hidden layers → output
        layer_sizes = [num_features] + hidden_units + [num_output]
        
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights_init = weights_init
        self.loss = loss
        self.optimizer = optimizer
        self.l2_coeff = l2_coeff
        self.dropout_p = dropout_p
        self.seed = seed
        
        # Validate dropout_p if provided
        num_hidden = len(hidden_units)
        if dropout_p is not None:
            if len(dropout_p) != num_hidden:
                raise ValueError(f"dropout_p must have {num_hidden} values (one per hidden layer)")
            self.dropout_p = dropout_p
        else:
            self.dropout_p = [0.0] * num_hidden  # No dropout by default
        
        # Initialize weights for each layer
        self.W = []
        key = random.PRNGKey(seed)
        
        for i in range(len(layer_sizes) - 1):
            key, subkey = random.split(key)
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            # Weight initialization
            if weights_init == 'he':
                # He initialization (good for ReLU)
                w = random.normal(subkey, (input_size + 1, output_size)) * jnp.sqrt(2 / input_size)
            elif weights_init == 'xavier':
                # Xavier initialization (good for tanh/sigmoid)
                w = random.normal(subkey, (input_size + 1, output_size)) * jnp.sqrt(1 / input_size)
            elif weights_init == 'normal':
                # Standard normal initialization
                w = random.normal(subkey, (input_size + 1, output_size)) * 0.01
            else:
                raise ValueError(f"Unknown weights_init: {weights_init}")
            
            self.W.append(w)
        
        # Initialize optimizer state
        if optimizer == 'adam':
            self.m = [jnp.zeros_like(w) for w in self.W]  # First moment estimates
            self.v = [jnp.zeros_like(w) for w in self.W]  # Second moment estimates
            self.t = 0  # Time step counter
        elif optimizer == 'rmsprop':
            self.v = [jnp.zeros_like(w) for w in self.W]  # Moving average of squared gradients


    def forward(self, X, W, dropout_on=False, rng_key=None):
        """
        Forward pass through the network with optional dropout

        Args:
            X: Input data
            W: Weights
            dropout_on: Whether to apply dropout (True during training, False during inference)
            rng_key: JAX random key for dropout (required if dropout_on=True)
        Returns:
            y: Output predictions
            h: List of hidden layer activations
            masks: List of dropout masks (one per hidden layer)
        """
        h = []
        masks = []
        a = X
        num_hidden = len(W) - 1
        
        # Loop through hidden layers
        for l in range(num_hidden):
            a = jnp.vstack([a, jnp.ones((1, a.shape[1]))])  # Add bias term
            z = W[l].T @ a
            a = self._activation_function(z)  # Use configurable activation
            
            # Apply dropout if enabled
            if dropout_on and self.dropout_p[l] > 0.0:
                if rng_key is None:
                    raise ValueError("rng_key must be provided when dropout_on=True")
                rng_key, subkey = random.split(rng_key)
                p = self.dropout_p[l]
                # Inverted dropout: scale active neurons to maintain expected activation
                mask = (random.uniform(subkey, a.shape) > p).astype(float) / (1.0 - p)
                a = a * mask
            else:
                mask = jnp.ones_like(a)  # No dropout: all neurons active
            
            h.append(a)
            masks.append(mask)
        
        # Output layer (no dropout)
        a = jnp.vstack([a, jnp.ones((1, a.shape[1]))])  # Add bias term
        y_hat = W[-1].T @ a
        y = self._softmax(y_hat)  # Output layer always uses softmax for classification
        return y, h, masks
    

    def backward(self, X, T, W, h, masks, eta, y_pred=None, use_clipping=True, max_grad_norm=25.0):
        """
        Backward pass with configurable optimizers, L2 regularization, gradient clipping, and dropout.
        
        Args:
            X: Input data
            T: Target labels
            W: Weights
            h: Hidden activations from forward pass
            masks: Dropout masks from forward pass
            eta: Learning rate
            y_pred: Pre-computed predictions (optional, for efficiency)
            use_clipping: Whether to use gradient clipping (default True)
            max_grad_norm: Maximum gradient norm for clipping (default 25.0)
        
        Returns:
            W: Updated weights
            loss: Loss value
            grad_norms: List of gradient norms per layer
        """
        m = X.shape[1]
        grad_norms = []  # Track gradient norms per layer
        
        if y_pred is None:  # Use pre-computed predictions if available, otherwise compute them
            y, _, _ = self.forward(X, W, dropout_on=False)
        else:
            y = y_pred
        
        # Increment Adam time step once per backward pass
        if self.optimizer == 'adam':
            self.t += 1
            
        delta = self._loss_derivative(y, T)  # Use configurable loss derivative
        
        # Backpropagate through hidden layers (in reverse)
        for l in range(len(W) - 1, 0, -1):
            a_prev = jnp.vstack([h[l-1], jnp.ones((1, h[l-1].shape[1]))])  # Add bias term
            Q = a_prev @ delta.T
            
            # Add L2 regularization to gradient (don't regularize biases - last row)
            if self.l2_coeff > 0:
                Q = Q.at[:-1, :].add(self.l2_coeff * W[l][:-1, :])  # Only regularize weights, not biases
            
            # Calculate gradient norm before clipping
            grad_norm = float(jnp.linalg.norm(Q))
            grad_norms.append(grad_norm)
            
            # Optional gradient clipping
            if use_clipping:
                Q = jnp.where(grad_norm > max_grad_norm, Q * (max_grad_norm / grad_norm), Q)
            
            # Apply optimizer update
            W = self._apply_optimizer_update(W, l, Q, eta, m)
            
            # Backpropagate delta
            delta = W[l][:-1, :] @ delta
            delta = delta * self._activation_derivative(h[l-1])  # Use configurable activation derivative
            delta = delta * masks[l-1]  # Apply dropout mask (only gradients through active neurons)
            
        # First layer gradient
        a_prev = jnp.vstack([X, jnp.ones((1, X.shape[1]))])  # Add bias term
        Q = a_prev @ delta.T
        
        # Add L2 regularization to first layer gradient
        if self.l2_coeff > 0:
            Q = Q.at[:-1, :].add(self.l2_coeff * W[0][:-1, :])  # Only regularize weights, not biases
        
        # Calculate gradient norm for first layer before clipping
        grad_norm = float(jnp.linalg.norm(Q))
        grad_norms.append(grad_norm)
        
        # Optional gradient clipping for first layer
        if use_clipping:
            Q = jnp.where(grad_norm > max_grad_norm, Q * (max_grad_norm / grad_norm), Q)
        
        # Apply optimizer update to first layer
        W = self._apply_optimizer_update(W, 0, Q, eta, m)
        loss = self._loss_function(y, T)
        
        # Reverse grad_norms to match layer order (0 to N)
        grad_norms.reverse()
        
        return W, loss, grad_norms


    def _apply_optimizer_update(self, W, layer_idx, gradients, eta, batch_size):
        """Apply optimizer-specific weight updates with optional update clipping."""
        # Create a copy of weights list for functional update
        W = [w for w in W]  # Shallow copy for JAX functional programming
        
                
        if self.optimizer == 'sgd':
            # Standard SGD update
            W[layer_idx] = W[layer_idx] - (eta / batch_size) * gradients
            
        elif self.optimizer == 'adam':
            # Adam optimizer with bias correction and update clipping
            beta1, beta2, epsilon = 0.9, 0.999, 1e-8
            # Update biased first moment estimate
            self.m[layer_idx] = beta1 * self.m[layer_idx] + (1 - beta1) * gradients
            # Update biased second raw moment estimate
            self.v[layer_idx] = beta2 * self.v[layer_idx] + (1 - beta2) * (gradients ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = self.m[layer_idx] / (1 - beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[layer_idx] / (1 - beta2 ** self.t)
            # Compute the raw update
            denominator = jnp.sqrt(v_hat) + epsilon
            update = (eta / batch_size) * m_hat / denominator
            update = update * (batch_size / 32)
            # Clip extreme updates to prevent instability
            update = jnp.clip(update, -1.0, 1.0)  # Element-wise clipping
            W[layer_idx] = W[layer_idx] - update
            
        elif self.optimizer == 'rmsprop':
            # RMSprop optimizer
            decay_rate, epsilon = 0.99, 1e-8
            # Update moving average of squared gradients
            self.v[layer_idx] = decay_rate * self.v[layer_idx] + (1 - decay_rate) * (gradients ** 2)
            # Apply update
            W[layer_idx] = W[layer_idx] - (eta / batch_size) * gradients / (jnp.sqrt(self.v[layer_idx]) + epsilon)

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        return W

    
    def _softmax(self, y_hat):
        """Compute softmax probabilities"""
        y_hat = y_hat - jnp.max(y_hat, axis=0, keepdims=True)  # prevent overflow
        exp_scores = jnp.exp(y_hat)
        return exp_scores / jnp.sum(exp_scores, axis=0, keepdims=True)
    

    def _activation_function(self, z):
        """Apply activation function"""
        if self.activation == 'relu':
            return jnp.maximum(0, z).astype(jnp.float32)
        elif self.activation == 'tanh':
            return jnp.tanh(z).astype(jnp.float32)
        elif self.activation == 'sigmoid':
            return (1 / (1 + jnp.exp(-jnp.clip(z, -500, 500)))).astype(jnp.float32)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        

    def _activation_derivative(self, a):
        """Calculate derivative of activation function"""
        if self.activation == 'relu':
            return (a > 0).astype(jnp.float32)
        elif self.activation == 'tanh':
            return (1 - a**2).astype(jnp.float32)
        elif self.activation == 'sigmoid':
            return (a * (1 - a)).astype(jnp.float32)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        

    def _loss_function(self, y_pred, y_true):
        """Calculate loss based on configured loss function"""
        epsilon = 1e-12  # Prevent log(0)
        
        if self.loss == 'cross_entropy':
            # Categorical Cross-Entropy Loss
            return -jnp.sum(jnp.log(jnp.sum(y_pred * y_true, axis=0) + epsilon))
        elif self.loss == 'mse':
            # Mean Squared Error Loss
            return 0.5 * jnp.sum((y_pred - y_true) ** 2)
        elif self.loss == 'mae':
            # Mean Absolute Error Loss
            return jnp.sum(jnp.abs(y_pred - y_true))
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
        

    def _loss_derivative(self, y_pred, y_true):
        """Calculate derivative of loss function for backpropagation"""
        if self.loss == 'cross_entropy':
            # For cross-entropy with softmax: derivative is simply (y_pred - y_true)
            return y_pred - y_true
        elif self.loss == 'mse':
            # MSE derivative: (y_pred - y_true)
            return y_pred - y_true
        elif self.loss == 'mae':
            # MAE derivative: sign(y_pred - y_true)
            return jnp.sign(y_pred - y_true)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")


# Shared utility functions
def calculate_accuracy(net, X, T, W):
    """Calculate accuracy percentage (always with dropout OFF)"""
    y, _, _ = net.forward(X, W, dropout_on=False)
    predictions = jnp.argmax(y, axis=0)
    true_labels = jnp.argmax(T, axis=0)
    return jnp.mean(predictions == true_labels) * 100
    

def train(net, X, T, W, epochs, eta, batchsize=32, X_val=None, T_val=None, use_clipping=True, max_grad_norm=25.0, use_wandb=False, wandb_project=None, wandb_config=None, wandb_mode="online"):
    """
    Training loop for neural network with mandatory validation.
    
    Args:
        net: Neural network instance
        X, T: Training data and labels
        W: Initial weights
        epochs: Number of training epochs
        eta: Learning rate
        batchsize: Mini-batch size
        X_val, T_val: Validation data and labels (required)
        use_clipping: Whether to use gradient clipping
        max_grad_norm: Maximum gradient norm for clipping
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_config: Dictionary of hyperparameters to log to W&B
        wandb_mode: W&B mode - "online", "offline", or "disabled"
    """
    losses = []
    val_losses = []  # Track validation loss
    train_accuracies = []  # Track training accuracy
    val_accuracies = []  # Track validation accuracy
    epoch_times = []  # Track computation time per epoch
    
    # Initialize W&B if enabled
    if use_wandb and wandb_project:
        wandb.init(project=wandb_project, config=wandb_config, mode=wandb_mode)
    
    # Print header for nicely formatted table
    print("-" * 90)
    print(f"{'Epoch':<10} {'Train Acc':<12} {'Val Acc':<12} {'Gain':<10} {'Time':<10} {'ETA'}")
    print("-" * 90)
    
    start_total = time.time()

    m = X.shape[1]  # Training set size
    m_val = X_val.shape[1]  # Validation set size
    key = random.PRNGKey(net.seed)  # For batch shuffling and dropout - uses network's seed
    
    for epoch in range(epochs):
        epoch_start = time.time()  # Start timing this epoch
        
        # Track gradient norms for this epoch
        epoch_grad_norms = []
        
        key, subkey = random.split(key)
        order = random.permutation(subkey, m)
        
        for i in range(0, m, batchsize):
            batch = order[i:i+batchsize]
            X_batch = X[:, batch]
            T_batch = T[:, batch]
            
            # Forward pass with dropout enabled during training
            key, dropout_key = random.split(key)
            y_batch, h, masks = net.forward(X_batch, W, dropout_on=True, rng_key=dropout_key)
            
            # Backward pass with dropout masks (this updates weights with dropout active)
            W, loss, grad_norms = net.backward(X_batch, T_batch, W, h, masks, eta, y_batch, use_clipping, max_grad_norm)
            epoch_grad_norms.append(grad_norms)

        # Calculate training accuracy and loss (with dropout OFF for fair comparison)
        train_accuracy = float(calculate_accuracy(net, X, T, W))
        train_accuracies.append(train_accuracy)
        # Calculate training loss with dropout OFF
        y_train_pred, _, _ = net.forward(X, W, dropout_on=False)
        train_loss = net._loss_function(y_train_pred, T)
        # Normalize training loss by dataset size (average loss per sample)
        losses.append(float(train_loss / m))
        
        # Calculate validation accuracy and loss
        val_accuracy = float(calculate_accuracy(net, X_val, T_val, W))
        val_accuracies.append(val_accuracy)
        # Calculate validation loss
        y_val_pred, _, _ = net.forward(X_val, W, dropout_on=False)
        val_loss = net._loss_function(y_val_pred, T_val)
        # Normalize validation loss by dataset size (average loss per sample)
        val_losses.append(float(val_loss / m_val))
        
        # Calculate average gradient norms across all batches in this epoch
        import numpy as np  # Use numpy for averaging lists
        avg_grad_norms = np.mean(epoch_grad_norms, axis=0)  # Average over batches
        total_grad_norm = float(jnp.linalg.norm(jnp.array(avg_grad_norms)))  # Total gradient norm
        
        # Calculate gain compared to last epoch
        if epoch > 0:
            gain = train_accuracy - train_accuracies[-2]  # Current - previous
            if gain > 0:
                gain_str = f"+{gain:.2f}%"
            elif gain < 0:
                gain_str = f"{gain:.2f}%"  # Already has negative sign
            else:
                gain_str = " 0.00%"
        else:
            gain_str = "baseline"
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Calculate ETA (estimated time remaining)
        if epoch > 0:
            avg_time_per_epoch = sum(epoch_times) / len(epoch_times)  # Use pure Python for averaging
            remaining_epochs = epochs - (epoch + 1)
            eta_seconds = avg_time_per_epoch * remaining_epochs
            if eta_seconds > 60:
                eta_str = f"{int(eta_seconds//60)}min {int(eta_seconds%60)}sec"
            else:
                eta_str = f"{int(eta_seconds)}sec"
        else:
            eta_str = "calculating..."
        
        # Format epoch info for output
        epoch_str = f"{epoch+1}/{epochs}"
        accuracy_str = f"{train_accuracy:.2f}%"
        val_acc_str = f"{val_accuracy:.2f}%"
        time_str = f"{epoch_time:.2f}sec"
        
        # Show progress 
        print(f"{epoch_str:<10} {accuracy_str:<12} {val_acc_str:<12} {gain_str:<10} {time_str:<10} {eta_str}")
        
        # Log to W&B if enabled
        if use_wandb and (wandb_project or wandb.run is not None):
            # Prepare log dictionary
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": float(losses[-1]),  # Use normalized loss
                "train_accuracy": float(train_accuracy),
                "val_accuracy": float(val_accuracy),
                "val_loss": float(val_losses[-1]),  # Use normalized loss
                "epoch_time": float(epoch_time),
                "grad_norm_total": float(total_grad_norm)  # Total gradient norm
            }
            
            # Log per-layer gradient norms
            for layer_idx, grad_norm in enumerate(avg_grad_norms):
                log_dict[f"grad_norm_layer_{layer_idx}"] = float(grad_norm)
            
            # Log parameter histograms every 1/10th of total epochs
            histogram_interval = max(1, epochs // 10)
            if (epoch + 1) % histogram_interval == 0 or epoch == 0:
                for layer_idx, w in enumerate(W):
                    # Separate weights and biases (bias is the last row)
                    weights = np.array(w[:-1, :])  # All rows except last
                    biases = np.array(w[-1, :])    # Last row
                    log_dict[f"weights_layer_{layer_idx}"] = wandb.Histogram(weights.flatten())
                    log_dict[f"biases_layer_{layer_idx}"] = wandb.Histogram(biases.flatten())
            
            wandb.log(log_dict)
    
    total_time = time.time() - start_total
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    print("-" * 70)
    print(f"Total training time: {total_time:.1f}sec")
    print(f"Average per epoch: {avg_epoch_time:.2f}sec")
    print("-" * 70)
    
    # Don't finish W&B here - let evaluate_model do it after logging test metrics
    
    return W, losses, train_accuracies, val_accuracies, val_losses

def evaluate_model(net, X_test, T_test, y_test, W, train_accuracies, use_wandb=False):
    """
    Evaluate model performance and print results
    
    Args:
        net: Neural network instance
        X_test, T_test: Test data and labels
        y_test: Test labels (not one-hot encoded)
        W: Trained weights
        train_accuracies: List of training accuracies from training
        use_wandb: Whether to log test metrics to W&B
    """
    # Make predictions and calculate accuracy (dropout OFF for evaluation)
    y_test_pred, _, _ = net.forward(X_test.T, W, dropout_on=False)
    y_pred = jnp.argmax(y_test_pred, axis=0)
    test_accuracy = float(jnp.mean(y_pred == y_test))  # Convert to Python float

    # Calculate test loss using the configurable loss function
    test_loss = float(net._loss_function(y_test_pred, T_test.T) / X_test.shape[0])  # Average per sample

    print(f"\n================== Final Results ==================")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss (avg per sample): {test_loss:.4f}")
    print(f"Training Accuracy Improvement: {(train_accuracies[-1] - train_accuracies[0]):.1f}% points")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    
    # Log test metrics to W&B if enabled and run is still active
    if use_wandb and wandb.run is not None:
        wandb.log({
            "test_accuracy": float(test_accuracy * 100),
            "test_loss": float(test_loss)
        })
        wandb.finish(quiet=False)  # Finish the W&B run after logging test metrics (quiet=False shows summary)
    
    return y_pred, test_accuracy, test_loss

def plot_training_results(losses, train_accuracies, val_accuracies, val_losses, test_accuracy=None, figsize=(15, 5), save_path=None):
    """
    Plot training curves including loss and accuracy over epochs.
    
    Args:
        losses: List of training loss values per epoch
        train_accuracies: List of training accuracy values per epoch
        val_accuracies: List of validation accuracy values per epoch (required)
        val_losses: List of validation loss values per epoch (required)
        test_accuracy: Final test accuracy (optional, shown as horizontal line)
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure (e.g., 'training_curves.png')
    
    Returns:
        fig: Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    epochs = range(1, len(losses) + 1)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Training and Validation Loss
    axes[0].plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    axes[0].plot(epochs, val_losses, 'orange', linestyle='--', linewidth=2, label='Validation Loss')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Loss Over Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Training and Validation Accuracy
    axes[1].plot(epochs, train_accuracies, 'g-', linewidth=2, label='Training Accuracy')
    axes[1].plot(epochs, val_accuracies, 'orange', linestyle='--', linewidth=2, label='Validation Accuracy')
    if test_accuracy is not None:
        axes[1].axhline(y=test_accuracy * 100, color='r', linestyle='--', linewidth=2, 
                       label=f'Test Accuracy: {test_accuracy * 100:.2f}%')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].set_title('Accuracy Over Time', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Accuracy Improvement (delta between consecutive epochs for train and val)
    train_deltas = [0] + [train_accuracies[i] - train_accuracies[i-1] 
                          for i in range(1, len(train_accuracies))]
    val_deltas = [0] + [val_accuracies[i] - val_accuracies[i-1] 
                       for i in range(1, len(val_accuracies))]
    
    # Bar width for side-by-side bars
    width = 0.5
    x = np.array(list(epochs))
    
    # Create bars
    train_colors = ['g' if d >= 0 else 'r' for d in train_deltas]
    val_colors = ['orange' if d >= 0 else 'darkred' for d in val_deltas]
    
    axes[2].bar(x - width/2, train_deltas, width, color=train_colors, alpha=0.6, label='Train')
    axes[2].bar(x + width/2, val_deltas, width, color=val_colors, alpha=0.6, label='Val')
    axes[2].legend()
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Accuracy Change (%)', fontsize=11)
    axes[2].set_title('Per-Epoch Accuracy Gain/Loss', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig



def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, figsize=(8, 6), save_path=None):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true: True labels (1D array of class indices)
        y_pred: Predicted labels (1D array of class indices)
        class_names: List of class names (optional, uses indices if None)
        normalize: Whether to normalize by row (True) or show raw counts (False)
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure (e.g., 'confusion_matrix.png')
    
    Returns:
        fig: Matplotlib figure object
        cm: Confusion matrix array
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    # Convert JAX arrays to numpy if needed
    if hasattr(y_true, 'device'):  # Check if it's a JAX array
        y_true = np.array(y_true)
    if hasattr(y_pred, 'device'):  # Check if it's a JAX array
        y_pred = np.array(y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Set up class names
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return fig, cm