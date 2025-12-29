# Standard library imports
import os

# Third-party library imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

# Local application imports
from config import TRANSFER_TARGET
from .plot import plot


def preparation(args):
    """
    Prepares the environment for training by setting up directories,
    choosing device (CPU or GPU), and logging output.

    Args:
        args: Command line arguments containing configurations.
    """
    # Create save directory path based on transfer target
    save_dir = os.path.join('saves', f'model_{TRANSFER_TARGET}')
    args.save_dir = save_dir

    # Create the directory if it does not exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Configure device based on availability and user preferences
    if torch.cuda.is_available() and args.devices:
        # Use specified CUDA devices
        args.device = [torch.device(f"cuda:{i}") for i in args.devices]
    else:
        # Default to CPU if no GPU is available
        args.device = torch.device("cpu")

    # Open log file for output recording
    args.output_file = open(os.path.join(args.save_dir, 'output.log'), 'a')

    # Log the arguments to file
    print(args, file=args.output_file, flush=True)


def eval(model, device, loader, criterion_fn):
    """
    Evaluates the model on the validation/test dataset.

    Args:
        model: The neural network model to evaluate.
        device: The device (CPU or GPU) to perform the evaluation.
        loader: DataLoader for the dataset to evaluate.
        criterion_fn: Loss function for evaluation.

    Returns:
        Tuple containing average loss, mean squared error (MSE), mean absolute error (MAE), and R² score.
    """
    # Set the model to evaluation mode
    model.eval()
    total_loss = 0
    y_true_list = []
    y_pred_list = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for data in loader:
            # Move data to device
            data = data.to(device)

            # Forward pass
            pred = model(data)

            # Calculate loss
            loss = criterion_fn(pred.view(-1), data.y.view(-1))
            total_loss += loss.item()

            # Collect true and predicted values
            y_true_list.append(data.y.view(-1).detach().cpu().numpy())
            y_pred_list.append(pred.view(-1).detach().cpu().numpy())

    # Concatenate all predictions and true values
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    # Calculate evaluation metrics
    avg_loss = total_loss / len(loader)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return avg_loss, mse, mae, r2


def augment_data(data, continuous_mask=None):
    """
    Enhanced molecular graph data augmentation with support for discrete/continuous feature separation.

    Args:
        data: Graph data object containing node features, edge attributes, etc.
        continuous_mask: Boolean mask indicating which node features are continuous.

    Returns:
        Augmented data object.
    """
    # Node feature perturbation
    if continuous_mask is None:
        # Default: treat all features as continuous
        noise = torch.randn_like(data.x) * 0.05
        data.x = data.x + noise
    else:
        # Add noise only to continuous features
        cont_features = data.x[:, continuous_mask]
        noise = torch.randn_like(cont_features) * 0.05
        data.x[:, continuous_mask] = cont_features + noise

        # Apply random masking to discrete features
        disc_features = data.x[:, ~continuous_mask]
        disc_mask = torch.rand(disc_features.shape) < 0.1
        disc_features[disc_mask] = 0
        data.x[:, ~continuous_mask] = disc_features

    # Edge feature perturbation
    if data.edge_attr is not None:
        edge_noise = torch.randn_like(data.edge_attr) * 0.03
        data.edge_attr = data.edge_attr + edge_noise

    return data


def smooth_labels(labels, factor=0.1, num_classes=None):
    """
    Enhanced label smoothing with support for multi-class and dynamic priors.

    Args:
        labels: Target labels to be smoothed.
        factor: Smoothing factor (0 = no smoothing, 1 = uniform distribution).
        num_classes: Number of classes for multi-class tasks.

    Returns:
        Smoothed labels.
    """
    if num_classes is None:
        # Regression or binary classification task
        prior = torch.tensor(0.5, device=labels.device)
    else:
        # Multi-class task uses uniform prior
        prior = torch.tensor(1.0 / num_classes, device=labels.device)

    return labels * (1 - factor) + prior * factor


def train(model: nn.Module,
          device: torch.device,
          loader: DataLoader,
          optimizer: optim.Optimizer,
          criterion_fn: nn.Module,
          l1_lambda: float,
          l2_lambda: float,
          scheduler: optim.lr_scheduler._LRScheduler = None,
          use_amp: bool = True,
          num_classes: int = None,
          continuous_mask: torch.Tensor = None) -> float:
    """
    Enhanced training function with:
    - Optimized molecular data augmentation
    - Dynamic label smoothing
    - Optimized regularization computation
    - Multi-device support
    - Learning rate scheduling

    Args:
        model: Neural network model to train.
        device: Target device (CPU/GPU/MPS).
        loader: Training data loader.
        optimizer: Optimizer for model parameters.
        criterion_fn: Loss function.
        l1_lambda: L1 regularization coefficient.
        l2_lambda: L2 regularization coefficient.
        scheduler: Learning rate scheduler (optional).
        use_amp: Enable mixed precision training.
        num_classes: Number of classes for classification tasks (for label smoothing).
        continuous_mask: Boolean mask indicating continuous nature of node features.

    Returns:
        Average loss value for the epoch.
    """
    # Set model to training mode
    model.train()
    total_loss = 0
    valid_batches = 0

    # Freeze batch normalization layers during training
    def freeze_bn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.InstanceNorm1d)):
            m.eval()

    if model.training:
        model.apply(freeze_bn)

    # Configure mixed precision training based on device support
    supported_amp_devices = ['cuda']
    use_amp_here = use_amp and (device.type in supported_amp_devices)
    scaler = torch.amp.GradScaler(enabled=use_amp_here)

    # Training loop
    for data in loader:
        # Move data to device
        data = data.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Apply data augmentation during training
        if model.training:
            data = augment_data(data, continuous_mask)

        # Mixed precision forward pass
        with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp_here
        ):
            # Forward pass
            pred = model(data)

            # Apply dynamic label smoothing
            smoothed_y = smooth_labels(
                data.y.view(-1),
                factor=0.1,
                num_classes=num_classes
            )

            # Calculate primary loss
            loss = criterion_fn(pred.view(-1), smoothed_y)

            # Efficient regularization computation
            reg_loss = 0
            for param in model.parameters():
                reg_loss += l1_lambda * torch.norm(param, 1)
                reg_loss += l2_lambda * torch.norm(param, 2)

            # Add regularization to total loss
            loss += reg_loss

        # Skip invalid batches (NaN or Inf loss)
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # Mixed precision backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            norm_type=2.0
        )

        # Optimizer update
        scaler.step(optimizer)
        scaler.update()

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        # Accumulate loss
        total_loss += loss.item()
        valid_batches += 1

    # Handle case where all batches are invalid
    if valid_batches == 0:
        return float('nan')

    return total_loss / valid_batches


def test(model, device, loader, fold):
    """
    Test the model and save evaluation results.

    Args:
        model: Trained neural network model.
        device: Device to run the test on.
        loader: Test data loader.
        fold: Current fold number for cross-validation.

    Returns:
        Tuple containing true values, predictions, MAE, R², and RMSE.
    """
    # Set model to evaluation mode
    model.eval()
    y_true = []
    y_pred = []

    # Disable gradient computation for testing
    with torch.no_grad():
        for data in loader:
            # Move data to device
            data = data.to(device)

            # Forward pass
            out = model(data)

            # Collect predictions and true values
            y_true.append(data.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())

    # Concatenate all results
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2'],
        'Value': [rmse, mae, r2]
    })

    # Save results to CSV
    results_df.to_csv(f'Output/GAT_model/{TRANSFER_TARGET}/{TRANSFER_TARGET}_evaluation_metrics_{fold + 1}.csv',
                      index=False)

    # Generate and save plots
    plot(y_true, y_pred, mae, r2, rmse, TRANSFER_TARGET, fold)

    return y_true, y_pred, mae, r2, rmse


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting by monitoring validation loss.
    """

    def __init__(self, patience=100, delta=0.001, path='checkpoint.pth'):
        """
        Initialize early stopping parameters.

        Args:
            patience (int): Number of epochs to wait for improvement in validation loss.
            delta (float): Minimum threshold for significant change in validation loss.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        """
        Check if early stopping should be triggered.

        Args:
            val_loss: Current validation loss.
            model: Model to potentially save.
        """
        if self.best_score is None:
            # First epoch - set initial best score
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            # Validation loss not improving
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Validation loss improved
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save the current best model.

        Args:
            val_loss: Current validation loss.
            model: Model to save.
        """
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def ensemble_predict(models, device, loader):
    """
    Perform ensemble prediction using multiple models.

    Args:
        models: List of trained models.
        device: Computation device.
        loader: Data loader for predictions.

    Returns:
        y_true: True values.
        y_pred_ensemble: Ensemble prediction values.
        results_df: Evaluation results DataFrame.
    """
    # Store predictions from each model
    all_preds = []
    y_true_list = []

    # Get predictions from each model
    for model in models:
        model.eval()
        fold_preds = []
        fold_true = []

        with torch.no_grad():
            for data in loader:
                # Move data to device
                data = data.to(device)

                # Forward pass
                out = model(data)

                # Collect predictions and true values
                fold_preds.append(out.cpu().numpy())
                fold_true.append(data.y.cpu().numpy())

        # Store predictions for this model
        all_preds.append(np.concatenate(fold_preds, axis=0))

        # Only need true values once
        if len(y_true_list) == 0:
            y_true_list = fold_true

    # Convert true values
    y_true = np.concatenate(y_true_list, axis=0)

    # Stack predictions (n_models, n_samples)
    all_preds = np.array(all_preds)

    # Calculate median as ensemble prediction
    y_pred_ensemble = np.median(all_preds, axis=0)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_true, y_pred_ensemble)
    r2 = r2_score(y_true, y_pred_ensemble)
    rmse = np.sqrt(np.mean((y_true - y_pred_ensemble) ** 2))

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2'],
        'Value': [rmse, mae, r2]
    })

    return y_true, y_pred_ensemble, results_df