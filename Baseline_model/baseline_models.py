# Standard library imports
import os

# Third-party scientific computing imports
import numpy as np
import pandas as pd

# Machine learning and data processing imports
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Deep learning imports
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Visualization imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Chemistry and molecular processing imports
from rdkit import Chem

# Progress bar import
from tqdm import tqdm

# Local imports
from Feature_calculation import (
    atom_id_names,
    bond_id_names,
    calculate_descriptors,
    get_column_molecules_smiles,
    mol_to_geognn_graph_data_MMFF3d,
    obtain_3D_mol
)
from config import RANDOM_SEED, TRANSFER_TARGET


def create_output_directories():
    """
    Create necessary output directories for storing baseline model results.

    Creates the following directory structure:
    - Output/Baseline/{TRANSFER_TARGET}/
    - Output/Baseline/{TRANSFER_TARGET}/dataset/
    - Output/Baseline/{TRANSFER_TARGET}/pics/
    """
    # Create main directory for output files specific to the transfer target
    os.makedirs(f"Output/Baseline/{TRANSFER_TARGET}", exist_ok=True)
    # Create subdirectory for dataset files
    os.makedirs(f"Output/Baseline/{TRANSFER_TARGET}/dataset", exist_ok=True)
    # Create subdirectory for visualization files
    os.makedirs(f"Output/Baseline/{TRANSFER_TARGET}/pics", exist_ok=True)


def save_3D_mol(all_smile, mol_save_dir):
    """
    Generate and save 3D molecular structures for a list of SMILES strings.

    This function processes each SMILES string to generate 3D molecular conformations
    using RDKit and saves them as MOL files. Failed conformations are tracked.

    Args:
        all_smile (list): List of SMILES strings representing molecular structures
        mol_save_dir (str): Directory path to save the 3D molecular structure files

    Returns:
        list: Indices of SMILES strings that failed to generate valid 3D structures
    """
    index = 0
    error_conformer = []  # Track failed conformer generations
    pbar = tqdm(all_smile)  # Progress bar for processing

    # Create output directory if it doesn't exist
    try:
        os.makedirs(f'{mol_save_dir}')
    except OSError:
        pass  # Directory already exists

    # Process each SMILES string
    for smiles in pbar:
        try:
            # Generate and save 3D molecular structure
            obtain_3D_mol(smiles, f'{mol_save_dir}/3D_mol_{index}')
        except ValueError:
            # Track failed conformer generation
            error_conformer.append(index)
            index += 1
            continue
        index += 1

    return error_conformer


def save_dataset(orderly_smile, mol_save_dir, orderly_name, descriptors_name, error_conformer, transfer_target):
    """
    Process molecular data and save as datasets for machine learning.

    This function loads 3D molecular structures, calculates molecular descriptors,
    generates graph representations, and saves the processed data for model training.

    Args:
        orderly_smile (list): List of SMILES strings in order
        mol_save_dir (str): Directory containing saved 3D molecular structures
        orderly_name (str): Output filename for the graph dataset
        descriptors_name (str): Output filename for the molecular descriptors
        error_conformer (list): Indices of molecules that failed 3D structure generation
        transfer_target (str): Target column type for descriptor calculation

    Returns:
        list: Indices of molecules that failed processing (same as error_conformer)
    """
    dataset = []  # Store graph data representations
    dataset_descriptors = []  # Store molecular descriptors
    pbar = tqdm(orderly_smile)  # Progress bar for processing
    index = 0

    for Smiles in pbar:
        # Skip molecules that failed 3D structure generation
        if index in error_conformer:
            index += 1
            continue

        # Load chiral molecule from MOL file
        mol_Chiral_path = os.path.join(mol_save_dir, f"Chiral_3D_mol/3D_mol_{index}.mol")
        if not os.path.isfile(mol_Chiral_path):
            raise FileNotFoundError(f"File not found: {mol_Chiral_path}")
        mol_Chiral = Chem.MolFromMolFile(mol_Chiral_path)
        if mol_Chiral is None:
            raise ValueError(f"Invalid MOL file: {mol_Chiral_path}")

        # Load column molecule from MOL file
        mol_Column_path = os.path.join(mol_save_dir, f"3D_mol_0.mol")
        if not os.path.isfile(mol_Column_path):
            raise FileNotFoundError(f"File not found: {mol_Column_path}")
        mol_Column = Chem.MolFromMolFile(mol_Column_path)
        if mol_Column is None:
            raise ValueError(f"Invalid MOL file: {mol_Chiral_path}")

        # Calculate molecular descriptors for both chiral and column molecules
        descriptor_Chiral = calculate_descriptors(mol_Chiral, transfer_target, 'chiral')
        descriptor_Column = calculate_descriptors(mol_Column, transfer_target, 'column')

        # Combine descriptors from chiral and column molecules
        combined_descriptor = np.concatenate([descriptor_Chiral, descriptor_Column])
        dataset_descriptors.append(combined_descriptor)

        # Generate graph data representation for the chiral molecule
        data = mol_to_geognn_graph_data_MMFF3d(mol_Chiral, transfer_target)
        dataset.append(data)

        index += 1

    # Save processed dataset and descriptors as numpy arrays
    dataset_descriptors = np.array(dataset_descriptors)
    np.save(f"{orderly_name}.npy", dataset, allow_pickle=True)
    np.save(f'{descriptors_name}.npy', dataset_descriptors)

    return error_conformer


def construct_dataset(dataset, RT, temperature_program, descriptor, column):
    """
    Construct input features and target values for machine learning models.

    This function processes molecular graph data, descriptors, and experimental conditions
    to create feature vectors suitable for regression models.

    Args:
        dataset (list): List of molecular graph data objects
        RT (list): List of retention time values (target variable)
        temperature_program (list): List of GC temperature program parameters
        descriptor (list): List of molecular descriptor arrays
        column (str): Column type identifier

    Returns:
        tuple: (data_x, data_y) where data_x contains input features and data_y contains target values
    """
    data_x = []  # Input features
    data_y = []  # Target values (retention times)

    for i in range(len(dataset)):
        data = dataset[i]
        tp = temperature_program[i]
        des = descriptor[i]

        # Process atomic features from graph data
        atom_x = []
        atom_edge_attr = []

        # Extract atom features using predefined atom ID names
        for name in atom_id_names:
            atom_x.append(data[name])

        # Extract bond features using predefined bond ID names
        for name in bond_id_names:
            atom_edge_attr.append(data[name])

        # Convert atom features to tensor and add continuous features
        atom_x = torch.from_numpy(np.array(atom_x).T).to(torch.int64)
        atom_float_feature_mass = torch.from_numpy(data["mass"].astype(np.float32))
        atom_float_feature_van = torch.from_numpy(data["van_der_waals_radis"].astype(np.float32))
        atom_x = torch.cat([atom_x, atom_float_feature_mass.reshape(-1, 1)], dim=1)
        atom_x = torch.cat([atom_x, atom_float_feature_van.reshape(-1, 1)], dim=1)

        # Aggregate atom features by summing
        atom_x = torch.sum(atom_x, dim=0, keepdim=True)
        atom_x = torch.flatten(atom_x)

        # Process bond features and add continuous features
        atom_edge_attr = torch.from_numpy(np.array(atom_edge_attr).T).to(torch.int64)
        bond_float_feature = torch.from_numpy(data['bond_length'].astype(np.float32))
        atom_edge_attr = torch.cat([atom_edge_attr, bond_float_feature.reshape(-1, 1)], dim=1)

        # Aggregate bond features by summing
        atom_edge_attr = torch.sum(atom_edge_attr, dim=0, keepdim=True)
        atom_edge_attr = torch.flatten(atom_edge_attr)

        # Convert descriptors and temperature program to tensors
        des = torch.tensor([des], dtype=torch.float32).view(-1)
        tp = torch.tensor([tp], dtype=torch.float32).view(-1)

        # Combine all edge-level features (bonds, descriptors, temperature program)
        edge_x = torch.cat([atom_edge_attr.view(-1), des, tp], dim=0)

        # Combine atom and edge features to create final input vector
        x = torch.cat([atom_x.view(-1), edge_x], dim=0)

        # Create target tensor
        y = torch.Tensor([RT[i]])

        data_x.append(x)
        data_y.append(y)

    return data_x, data_y


def process_dataset():
    """
    Process the raw dataset and prepare it for machine learning.

    This function loads the complete dataset, filters it for the specific transfer target,
    extracts molecular information, generates 3D structures, and saves processed data.
    """
    # Load the complete dataset from CSV file
    dataset_all = pd.read_csv('dataset/data.csv')

    # Filter dataset for the specific transfer target (column type)
    dataset_target = dataset_all[dataset_all['Column'] == TRANSFER_TARGET]

    # Save the filtered dataset for this transfer target
    dataset_target.to_csv(f"Output/Baseline/{TRANSFER_TARGET}/dataset/{TRANSFER_TARGET}_orderly.csv")

    # Extract SMILES strings for chiral molecules from the filtered dataset
    Chiral_molecules_smiles = dataset_target['Chiral_molecules_smile'].values

    # Get SMILES string for the column molecule
    Column_molecules_smiles = [get_column_molecules_smiles(TRANSFER_TARGET)]

    # Extract temperature program data (columns 4-18 contain GC parameters)
    temperature_program = dataset_all.iloc[:, 4:18]

    # Save temperature program data
    np.save(f'Output/Baseline/{TRANSFER_TARGET}/dataset/{TRANSFER_TARGET}_temperature_program.npy', temperature_program)

    # Generate and save 3D molecular structures, track failed conformations
    error_indices = save_3D_mol(Chiral_molecules_smiles, f'Output/Baseline/{TRANSFER_TARGET}/Chiral_3D_mol')
    Column_molecules = save_3D_mol(Column_molecules_smiles, f'Output/Baseline/{TRANSFER_TARGET}')

    # Print and save indices of molecules that failed 3D structure generation
    print(f'error_{TRANSFER_TARGET}:', error_indices)
    np.save(f'Output/Baseline/{TRANSFER_TARGET}/dataset/error_{TRANSFER_TARGET}.npy', np.array(error_indices))

    # Process and save the complete dataset with descriptors and graph representations
    save_dataset(
        Chiral_molecules_smiles,
        f'Output/Baseline/{TRANSFER_TARGET}',
        f'Output/Baseline/{TRANSFER_TARGET}/dataset/dataset_{TRANSFER_TARGET}_orderly',
        f'Output/Baseline/{TRANSFER_TARGET}/dataset/dataset_{TRANSFER_TARGET}_orderly_descriptors',
        error_indices,
        TRANSFER_TARGET
    )


def load_dataset():
    """
    Load and preprocess the saved dataset for model training.

    This function loads the processed dataset, removes molecules with failed conformations,
    and returns all necessary components for machine learning.

    Returns:
        tuple: (gc, descriptor, all_smile, rt, dataset, temperature_program)
            - gc: DataFrame with cleaned molecular data
            - descriptor: Array of molecular descriptors
            - all_smile: Array of SMILES strings
            - rt: Array of retention time values
            - dataset: List of graph data objects
            - temperature_program: Array of GC temperature parameters
    """
    # Load the processed dataset CSV file
    gc = pd.read_csv(f'Output/Baseline/{TRANSFER_TARGET}/dataset/{TRANSFER_TARGET}_orderly.csv')

    # Load indices of molecules with failed 3D structure generation
    bad_index = np.load(f'Output/Baseline/{TRANSFER_TARGET}/dataset/error_{TRANSFER_TARGET}.npy')

    # Remove molecules with failed conformations and reset index
    gc = gc.drop(bad_index).reset_index(drop=True)

    # Load precomputed molecular descriptors
    descriptor = np.load(f'Output/Baseline/{TRANSFER_TARGET}/dataset/dataset_{TRANSFER_TARGET}_orderly_descriptors.npy')

    # Load GC temperature program parameters
    temperature_program = np.load(
        f'Output/Baseline/{TRANSFER_TARGET}/dataset/{TRANSFER_TARGET}_temperature_program.npy')

    # Extract SMILES strings and retention time values
    all_smile = gc['Chiral_molecules_smile'].values
    rt = gc['RT'].values

    # Load the graph dataset with molecular representations
    dataset = np.load(f'Output/Baseline/{TRANSFER_TARGET}/dataset/dataset_{TRANSFER_TARGET}_orderly.npy',
                      allow_pickle=True).tolist()

    return gc, descriptor, all_smile, rt, dataset, temperature_program


def prepare_data(dataset, rt, temperature_program, descriptor):
    """
    Prepare data splits for model training, validation, and testing.

    This function constructs the final dataset and splits it into training, validation,
    and test sets with proper data format conversion for scikit-learn models.

    Args:
        dataset (list): List of molecular graph data objects
        rt (array): Array of retention time values
        temperature_program (array): Array of GC temperature parameters
        descriptor (array): Array of molecular descriptors

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
            Training, validation, and test sets for features and targets
    """
    # Construct the dataset for model training by combining all features
    x, y = construct_dataset(dataset, rt, temperature_program, descriptor, column=TRANSFER_TARGET)

    # Split data: 80% for training+validation, 20% for final testing
    X_temp, X_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)

    # Split the temporary set: 70% training, 10% validation (12.5% of temp = 10% of total)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=RANDOM_SEED)

    # Convert PyTorch tensors to NumPy arrays for scikit-learn compatibility
    X_train = torch.stack(X_train).numpy()
    X_val = torch.stack(X_val).numpy()
    X_test = torch.stack(X_test).numpy()

    # Remove extra dimension if present (squeeze batch dimension)
    if X_train.ndim == 3:
        X_train = X_train.squeeze(1)
        X_val = X_val.squeeze(1)
        X_test = X_test.squeeze(1)

    # Convert target tensors to flat NumPy arrays
    y_train = torch.stack(y_train).numpy().ravel()
    y_val = torch.stack(y_val).numpy().ravel()
    y_test = torch.stack(y_test).numpy().ravel()

    return X_train, X_val, X_test, y_train, y_val, y_test


class TransformerRegressor(nn.Module):
    """
    Transformer-based regression model for retention time prediction.

    This class implements a Transformer encoder architecture adapted for regression tasks.
    It uses self-attention mechanisms to capture relationships in molecular features.
    """

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        """
        Initialize the Transformer regression model.

        Args:
            input_dim (int): Dimension of input features
            d_model (int): Dimension of the model (embedding size)
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate for regularization
        """
        super().__init__()

        # Linear layer to project input features to model dimension
        self.embedding = nn.Linear(input_dim, d_model)

        # Transformer encoder layer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Stack multiple transformer encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection layers with regularization
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )

    def forward(self, x):
        """
        Forward pass through the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Predicted retention times of shape (batch_size,)
        """
        # Add sequence dimension (batch_size, seq_len=1, features)
        x = x.unsqueeze(1)

        # Project input to model dimension
        embedded = self.embedding(x)

        # Apply transformer layers
        transformed = self.transformer(embedded)

        # Global average pooling across sequence dimension
        pooled = transformed.mean(dim=1)

        # Final prediction through output layers
        return self.fc_out(pooled).squeeze(1)


def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate multiple regression models on the dataset.

    This function trains various machine learning models including traditional ML algorithms
    and deep learning models, then evaluates their performance on validation and test sets.

    Args:
        X_train, X_val, X_test (np.ndarray): Feature arrays for training, validation, and testing
        y_train, y_val, y_test (np.ndarray): Target arrays for training, validation, and testing

    Returns:
        list: List of dictionaries containing evaluation results for each model
    """
    # Dictionary of regression models with optimized hyperparameters
    models = {
        # Linear regression with basic configuration
        'Linear_Regression': LinearRegression(
            fit_intercept=True,
            copy_X=True,
            n_jobs=-1  # Use all available processors
        ),

        # Support Vector Regression with RBF kernel
        'Support_Vector_Regression': SVR(
            kernel='rbf',
            C=100,  # Regularization parameter
            epsilon=0.5,  # Epsilon-tube tolerance
            gamma=1,  # RBF kernel parameter
            tol=1e-5,  # Tolerance for stopping criterion
            cache_size=200,  # Kernel cache size in MB
            shrinking=True,
            verbose=False,
            max_iter=-1
        ),

        # Gradient Boosting with regularization
        'Gradient_Boosting_Regressor': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=5,
            subsample=0.8,  # Stochastic gradient boosting
            max_features='sqrt',  # Random feature selection
            n_iter_no_change=100,  # Early stopping
            validation_fraction=0.5,
            tol=1e-3,
            random_state=42
        ),

        # Multi-layer Perceptron with adaptive learning
        'Neural_Network_Regressor': MLPRegressor(
            hidden_layer_sizes=(512, 128, 32),  # Three hidden layers
            activation='relu',
            solver='adam',
            alpha=0.01,  # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=100,
            random_state=42
        ),

        # Transformer model for deep learning approach
        'Transformer': TransformerRegressor(
            input_dim=X_train.shape[1],
            d_model=32,
            nhead=4,
            num_layers=6,
            dim_feedforward=8,
            dropout=0.1
        ),

        # Random Forest with ensemble learning
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=5,
            max_features='sqrt',  # Random feature selection
            bootstrap=True,  # Bootstrap sampling
            oob_score=True,  # Out-of-bag score estimation
            n_jobs=-1,  # Use all processors
            random_state=42
        )
    }

    # Create directory for saving trained models
    os.makedirs(f'saves/{TRANSFER_TARGET}', exist_ok=True)
    results = []

    # Train and evaluate each model
    for name, model in models.items():
        if name == 'Transformer':
            # Special handling for Transformer model (PyTorch-based)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # Prepare data loaders for mini-batch training
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32)
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            # Training configuration with advanced optimizers
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            criterion = nn.MSELoss()

            # Training loop with early stopping
            best_val_loss = float('inf')
            patience = 100
            patience_counter = 0

            for epoch in range(5000):
                # Training phase
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                # Validation phase
                model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                    y_pred_val = model(X_val_t).cpu().numpy()
                    val_loss = mean_squared_error(y_val, y_pred_val)
                    scheduler.step(val_loss)

                # Early stopping mechanism
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'saves/{TRANSFER_TARGET}_Baseline/{name}_best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Load best model and make test predictions
            model.load_state_dict(torch.load(
                f'saves/{TRANSFER_TARGET}_Baseline/{name}_best_model.pth',
                weights_only=True))
            model.eval()
            with torch.no_grad():
                X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_pred_test = model(X_test_t).cpu().numpy().flatten()
        else:
            # Standard scikit-learn model training
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)

        # Calculate comprehensive evaluation metrics for validation set
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_mse = mean_squared_error(y_val, y_pred_val)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(y_val, y_pred_val)

        # Calculate comprehensive evaluation metrics for test set
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_pred_test)

        # Store comprehensive results for comparison
        results.append({
            'Model': name,
            'Validation RMSE': val_rmse,
            'Validation MAE': val_mae,
            'Validation R2': val_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'Test R2': test_r2,
        })

        # Generate visualization plots
        plot(y_test, y_pred_test, test_mae, test_r2, test_rmse, name)

        # Save predictions for further analysis
        save_predictions(y_test, y_pred_test, name)

        # Save the trained model for future use
        os.makedirs(f'saves/{TRANSFER_TARGET}_Baseline/', exist_ok=True)
        joblib.dump(model, f'saves/{TRANSFER_TARGET}_Baseline/{name}_best_model.pkl')
        print(f'model: {name} with MSE: {test_mse:.4f}')

    return results


def save_predictions(y_true, y_pred, model_name):
    """
    Save model predictions to CSV file for analysis.

    Args:
        y_true (array): Actual retention time values
        y_pred (array): Predicted retention time values
        model_name (str): Name of the model for file naming
    """
    # Create DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        'Actual RT': y_true,
        'Predicted RT': y_pred
    })

    # Save to CSV file in the output directory
    results_df.to_csv(f'Output/Baseline/{TRANSFER_TARGET}/{model_name}_predictions.csv', index=False)


def validate_external_dataset(external_dataset_path, name):
    """
    Validate trained models on an external dataset.

    This function loads an external dataset, processes it using the same pipeline
    as the training data, and evaluates the saved best model's performance.

    Args:
        external_dataset_path (str): Path to the external validation dataset
        name (str): Name of the model to validate
    """
    # Load external validation dataset
    external_dataset = pd.read_csv(external_dataset_path)

    # Filter dataset for the specific transfer target (same column type)
    dataset_target = external_dataset[external_dataset['Column'] == f'{TRANSFER_TARGET}']

    # Create output directories for external validation results
    os.makedirs(f"Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}", exist_ok=True)
    os.makedirs(f"Validate_External/results/Baseline_model/{TRANSFER_TARGET}", exist_ok=True)

    # Save the filtered external dataset
    dataset_target.to_csv(
        f"Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/{TRANSFER_TARGET}_orderly.csv")

    # Extract molecular information from external dataset
    Chiral_molecules_smiles = dataset_target['Chiral_molecules_smile'].values
    Column_molecules_smiles = [get_column_molecules_smiles(TRANSFER_TARGET)]

    # Extract and save temperature program data
    temperature_program = external_dataset.iloc[:, 4:18]
    np.save(f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/external_temperature_program.npy',
            temperature_program)

    # Generate 3D structures for external molecules and track failures
    error_smile = save_3D_mol(Chiral_molecules_smiles,
                              f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/Chiral_3D_mol')
    Column_molecules = save_3D_mol(Column_molecules_smiles,
                                   f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}')

    print(f'error_{TRANSFER_TARGET}:', error_smile)

    # Save error indices for external dataset
    np.save(f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/error_{TRANSFER_TARGET}.npy',
            np.array(error_smile))

    # Process external dataset using the same pipeline
    save_dataset(
        Chiral_molecules_smiles,
        f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}',
        f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/external_dataset',
        f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/external_dataset_descriptors',
        error_smile,
        TRANSFER_TARGET
    )

    # Load the trained model for validation
    best_model = joblib.load(f'saves/{TRANSFER_TARGET}_Baseline/{name}_best_model.pkl')

    # Prepare external data using the same preprocessing pipeline
    gc = pd.read_csv(f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/{TRANSFER_TARGET}_orderly.csv')
    bad_index = np.load(f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/error_{TRANSFER_TARGET}.npy')
    gc = gc.drop(bad_index).reset_index(drop=True)  # Remove molecules with conformer errors

    # Load processed external data components
    descriptor = np.load(
        f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/external_dataset_descriptors.npy')
    temperature_program = np.load(
        f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/external_temperature_program.npy')
    all_smile = gc['Chiral_molecules_smile'].values
    rt = gc['RT'].values
    dataset = np.load(f'Validate_External/external_data/Baseline_model/{TRANSFER_TARGET}/external_dataset.npy',
                      allow_pickle=True).tolist()

    # Construct feature vectors using the same function as training
    x_external, _ = construct_dataset(dataset, rt, temperature_program, descriptor,
                                      column=TRANSFER_TARGET)

    # Convert to numpy arrays for model prediction
    X_external = torch.stack(x_external).numpy()
    if X_external.ndim == 3:
        X_external = X_external.squeeze(1)

    # Make predictions using the appropriate model type
    if name == 'Transformer':
        # Special handling for Transformer model
        model = TransformerRegressor(input_dim=X_external.shape[1])
        model.load_state_dict(torch.load(
            f'saves/{TRANSFER_TARGET}_Baseline/{name}_best_model.pth',
            weights_only=True))
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        with torch.no_grad():
            X_external_t = torch.tensor(X_external, dtype=torch.float32).to(device)
            y_external_pred = model(X_external_t).cpu().numpy().flatten()
    else:
        # Use scikit-learn model for prediction
        y_external_pred = best_model.predict(X_external)

    # Save prediction results to CSV
    results_df = pd.DataFrame({
        'Actual RT': rt,
        'Predicted RT': y_external_pred,
    })
    results_df.to_csv(f'Validate_External/results/Baseline_model/{TRANSFER_TARGET}/{name}_predictions_VE.csv',
                      index=False)

    # Calculate evaluation metrics for external validation
    mse = mean_squared_error(rt, y_external_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(rt, y_external_pred)  # Mean Absolute Error

    # Create DataFrame to store evaluation metrics
    results_df_metrics = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE'],  # List of evaluation metrics
        'Value': [mse, rmse, mae]  # Corresponding metric values
    })

    # Print and save evaluation results
    print(results_df_metrics)
    results_df_metrics.to_csv(
        f'Validate_External/results/Baseline_model/{TRANSFER_TARGET}/{name}_external_evaluation_metrics_VE.csv',
        index=False)

    print('External dataset validation completed and predictions saved.')


def plot(y_true, y_pred, mae, r2, rmse, model_name):
    """
    Generate visualization plots for model predictions.

    This function creates two types of plots:
    1. Scatter plot (hexbin) showing predicted vs actual values with performance metrics
    2. Histogram showing the distribution of prediction errors

    Args:
        y_true (array): Actual retention time values
        y_pred (array): Predicted retention time values
        mae (float): Mean Absolute Error
        r2 (float): R-squared coefficient
        rmse (float): Root Mean Squared Error
        model_name (str): Name of the model for plot titles and file naming
    """
    # Uncomment these lines to set font family to Arial for publication quality
    # matplotlib.rcParams['font.family'] = 'sans-serif'
    # matplotlib.rcParams['font.sans-serif'] = ['Arial']

    # Initialize color scheme variables
    clist = []

    # Set color scheme based on the transfer target (different columns have different colors)
    if TRANSFER_TARGET == 'Cyclosil_B':
        clist = ['#ffffff', '#8fbbda', '#1f77b4']  # Blue color scheme
        facecolor = '#1f77b4'
        textcolor = '#1f77b4'

    elif TRANSFER_TARGET == 'Cyclodex_B':
        clist = ['#ffffff', '#ffbf87', '#ff7f0e']  # Orange color scheme
        facecolor = '#ff7f0e'
        textcolor = '#ff7f0e'

    elif TRANSFER_TARGET == 'HP_chiral_20β':
        clist = ['#ffffff', '#96d096', '#2ca02c']  # Green color scheme
        facecolor = '#2ca02c'
        textcolor = '#2ca02c'

    elif TRANSFER_TARGET == 'CP_Cyclodextrin_β_2,3,6_M_19':
        clist = ['#ffffff', '#eb9394', '#d62728']  # Red color scheme
        facecolor = '#d62728'
        textcolor = '#d62728'

    elif TRANSFER_TARGET == 'CP_Chirasil_D_Val':
        clist = ['#ffffff', '#cab3de', '#9467bd']  # Purple color scheme
        facecolor = '#9467bd'
        textcolor = '#9467bd'

    elif TRANSFER_TARGET == 'CP_Chirasil_Dex_CB':
        clist = ['#ffffff', '#c6aba5', '#8c564b']  # Brown color scheme
        facecolor = '#8c564b'
        textcolor = '#8c564b'

    elif TRANSFER_TARGET == 'CP_Chirasil_L_Val':
        clist = ['#ffffff', '#f1bbe1', '#e377c2']  # Pink color scheme
        facecolor = '#e377c2'
        textcolor = '#e377c2'

    # Create custom colormap from color list
    newcmp = LinearSegmentedColormap.from_list('chaos', clist)

    # Flatten prediction arrays for plotting
    out_y_pred = np.reshape(y_pred, (-1,))
    out_y_test = np.reshape(y_true, (-1,))

    # Determine axis limits based on data range
    xmin = out_y_test.min()
    xmax = out_y_test.max()

    # Create scatter plot (hexbin) for predicted vs actual values
    fig = plt.figure(figsize=(10, 6))
    plt.rcParams['xtick.direction'] = 'in'  # Tick marks point inward
    plt.rcParams['ytick.direction'] = 'in'

    # Set axis labels with formatting
    plt.xlabel('Real values for RT', fontsize=18, weight='bold')
    plt.ylabel('Predicted values for RT', fontsize=18, weight='bold')
    plt.yticks(size=16)
    plt.xticks(size=16)

    # Plot perfect prediction line (diagonal)
    plt.plot([xmin, xmax], [xmin, xmax], ':', linewidth=4, color='red')

    # Add performance metrics as text annotations
    plt.text(xmin + (xmax - xmin) * 0.02, xmax - (xmax - xmin) * 0.05, f'MAE: {mae:.2f}',
             weight='bold', fontsize=20, color=textcolor)
    plt.text(xmin + (xmax - xmin) * 0.02, xmax - (xmax - xmin) * 0.1, f'RMSE: {rmse:.2f}',
             weight='bold', fontsize=20, color=textcolor)
    plt.text(xmin + (xmax - xmin) * 0.02, xmax - (xmax - xmin) * 0.15, f'R²: {r2:.2f}',
             weight='bold', fontsize=20, color=textcolor)

    # Create hexagonal binning plot for density visualization
    plt.hexbin(out_y_test, out_y_pred, gridsize=20, extent=[xmin, xmax, xmin, xmax],
               cmap=newcmp)
    plt.axis([xmin, xmax, xmin, xmax])

    # Configure axis appearance
    ax = plt.gca()
    ax.tick_params(top=True, right=True)  # Show ticks on all sides

    # Add colorbar to show frequency scale
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Frequency', size=18, weight='bold', rotation=270, labelpad=15)

    # Save scatter plot with high resolution
    plt.savefig(f'Output/Baseline/{TRANSFER_TARGET}/pics/{model_name}_predictions.png', dpi=600)
    plt.close()

    # Create error distribution histogram
    fig = plt.figure(figsize=(8, 3))
    plt.subplots_adjust(bottom=0.25)  # Add space for labels

    # Calculate prediction errors
    errors = y_true - y_pred

    # Plot histogram of errors
    plt.hist(errors, bins=200, facecolor=facecolor, alpha=0.7)

    # Set labels and formatting
    plt.xlabel("Error", fontsize=18, weight='bold')
    plt.ylabel("Frequency", fontsize=18, weight='bold')
    plt.yticks(size=16)
    plt.xticks(size=16)

    # Save error histogram with high resolution
    plt.savefig(f'Output/Baseline/{TRANSFER_TARGET}/pics/{model_name}_Error.png', dpi=600)
    plt.close()