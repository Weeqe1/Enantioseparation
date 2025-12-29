# Standard library imports
import numpy as np
import pandas as pd

# Third-party library imports
import torch

# Local application imports
from config import TRANSFER_TARGET
from Feature_calculation import Construct_dataset


def load_external_data():
    """
    Loads and processes external data for model validation.

    This function reads external validation data from CSV files and numpy arrays,
    then constructs a graph dataset for model validation purposes.

    Returns:
        external_graph: A constructed dataset graph from the external data
                       containing molecular graphs, retention times, temperature
                       programs, and descriptors for the target column.
    """
    # Load external validation data from the specified CSV file
    # The file contains retention time data for the target column
    external_data = pd.read_csv(
        f'Output/Validate_External/external_data/{TRANSFER_TARGET}/{TRANSFER_TARGET}_orderly.csv')

    # Extract the retention time (RT) values from the external data
    # These will serve as the ground truth labels for validation
    external_RT = external_data['RT'].values

    # Load additional external datasets from numpy files
    # Molecular descriptors containing chemical features
    external_descriptor = np.load(
        f'Output/Validate_External/external_data/{TRANSFER_TARGET}/external_dataset_descriptors.npy')

    # Temperature program data for gas chromatography conditions
    external_temperature_program = np.load(
        f'Output/Validate_External/external_data/{TRANSFER_TARGET}/external_temperature_program.npy')

    # Dataset containing molecular structures and related information
    external_dataset = np.load(f'Output/Validate_External/external_data/{TRANSFER_TARGET}/external_dataset.npy',
                               allow_pickle=True).tolist()

    # Construct a graph dataset from the external data
    # This creates PyTorch Geometric compatible data objects
    external_graph = Construct_dataset(external_dataset, external_RT, external_temperature_program, external_descriptor,
                                       column=TRANSFER_TARGET)

    return external_graph


def Validate_External(model, device, external_loader):
    """
    Validates a trained model on external test data.

    This function evaluates the model performance on an external dataset
    by collecting predictions and ground truth values.

    Args:
        model: The trained PyTorch model to be validated
        device: The device (CPU/GPU) where the model and data should be placed
        external_loader: DataLoader containing the external validation dataset

    Returns:
        tuple: A tuple containing:
            - y_true (numpy.ndarray): Ground truth retention time values
            - y_pred (numpy.ndarray): Model predicted retention time values
    """
    # Set model to evaluation mode to disable dropout and batch normalization updates
    model.eval()

    # Initialize lists to store true and predicted values
    y_true = []
    y_pred = []

    # Disable gradient computation for inference to save memory and speed up computation
    with torch.no_grad():
        # Iterate through all batches in the external data loader
        for data in external_loader:
            # Move data to the specified device (CPU or GPU)
            data = data.to(device)

            # Forward pass: get model predictions
            pred = model(data)

            # Collect ground truth values (flatten to 1D)
            y_true.append(data.y.view(-1))

            # Collect predicted values (flatten to 1D)
            y_pred.append(pred.view(-1))

    # Concatenate all batches and convert to numpy arrays for metric calculation
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

    return y_true, y_pred