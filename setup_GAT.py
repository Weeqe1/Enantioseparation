"""
Main setup script for GAT (Graph Attention Network) model training and validation.

This script provides three main functionalities based on the DESK configuration:
1. Dataset construction and processing for GAT model
2. GAT model training with 5-fold cross-validation
3. External dataset validation using trained GAT model
"""

# Standard library imports
import os
import random
import shutil
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm

# Local imports - Feature calculation and dataset processing
from Feature_calculation import (
    Construct_dataset,
    get_column_molecules_smiles,
    save_3D_mol,
    save_dataset
)

# Local imports - GAT model and arguments
from GAT_model import GAT
from GAT_model.parse_args import parse_args

# Local imports - Training utilities
from train import (
    EarlyStopping,
    ensemble_predict,
    eval,
    plot,
    preparation,
    test,
    train
)

# Local imports - External validation
from Validate_External import Validate_External, load_external_data

# Local imports - Configuration
from config import DESK, RANDOM_SEED, TRANSFER_TARGET

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seeds for reproducibility across all libraries
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set PyTorch random seeds
seed = torch.Generator().manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Parse arguments and prepare training environment
args = parse_args(TRANSFER_TARGET)
preparation(args)

# Main execution logic based on DESK configuration
if DESK == 'Dataset_construct':
    """
    Dataset Construction Mode:
    - Loads and filters dataset for target column
    - Creates output directories for GAT model
    - Processes molecular structures and saves 3D molecules
    - Constructs and saves the complete dataset with descriptors
    """
    # Load complete dataset and filter for target column
    dataset_all = pd.read_csv('dataset/data.csv')
    dataset_target = dataset_all[dataset_all['Column'] == f'{TRANSFER_TARGET}']

    # Create output directories for GAT model results
    os.makedirs(f"Output/GAT_model/{TRANSFER_TARGET}", exist_ok=True)
    os.makedirs(f"Output/GAT_model/{TRANSFER_TARGET}/pics", exist_ok=True)
    os.makedirs(f"Output/GAT_model/{TRANSFER_TARGET}/dataset", exist_ok=True)

    # Save filtered dataset
    dataset_target.to_csv(f"Output/GAT_model/{TRANSFER_TARGET}/dataset/{TRANSFER_TARGET}_orderly.csv")

    # Extract molecular SMILES and column information
    Chiral_molecules_smiles = dataset_target['Chiral_molecules_smile'].values
    Column_molecules_smiles = [get_column_molecules_smiles(TRANSFER_TARGET)]

    # Extract and save temperature program data
    temperature_program = dataset_all.iloc[:, 4:18]
    np.save(f'Output/GAT_model/{TRANSFER_TARGET}/dataset/{TRANSFER_TARGET}_temperature_program.npy',
            temperature_program)

    # Generate and save 3D molecular structures
    error_smile = save_3D_mol(Chiral_molecules_smiles, f'Output/GAT_model/{TRANSFER_TARGET}/Chiral_3D_mol')
    Column_molecules = save_3D_mol(Column_molecules_smiles, f'Output/GAT_model/{TRANSFER_TARGET}')

    # Log and save error information
    print(f'error_{TRANSFER_TARGET}:', error_smile)
    np.save(f'Output/GAT_model/{TRANSFER_TARGET}/dataset/error_{TRANSFER_TARGET}.npy', np.array(error_smile))

    # Save complete dataset with descriptors
    save_dataset(
        Chiral_molecules_smiles,
        f'Output/GAT_model/{TRANSFER_TARGET}',
        f'Output/GAT_model/{TRANSFER_TARGET}/dataset/dataset_{TRANSFER_TARGET}_orderly',
        f'Output/GAT_model/{TRANSFER_TARGET}/dataset/dataset_{TRANSFER_TARGET}_orderly_descriptors',
        error_smile,
        TRANSFER_TARGET
    )

elif DESK == 'Train':
    """
    Training Mode:
    - Loads preprocessed dataset and removes error samples
    - Performs 5-fold cross-validation training
    - Trains GAT models with early stopping and learning rate scheduling
    - Evaluates models and saves best performing model
    - Performs ensemble prediction using all fold models
    """
    # Load preprocessed data components
    GC = pd.read_csv(f'Output/GAT_model/{TRANSFER_TARGET}/dataset/{TRANSFER_TARGET}_orderly.csv')
    bad_index = np.load(f'Output/GAT_model/{TRANSFER_TARGET}/dataset/error_{TRANSFER_TARGET}.npy')
    GC = GC.drop(bad_index).reset_index(drop=True)

    # Load molecular descriptors and experimental data
    descriptor = np.load(
        f'Output/GAT_model/{TRANSFER_TARGET}/dataset/dataset_{TRANSFER_TARGET}_orderly_descriptors.npy')
    RT = GC['RT'].values
    temperature_program = np.load(
        f'Output/GAT_model/{TRANSFER_TARGET}/dataset/{TRANSFER_TARGET}_temperature_program.npy')

    # Load graph dataset
    dataset = np.load(f'Output/GAT_model/{TRANSFER_TARGET}/dataset/dataset_{TRANSFER_TARGET}_orderly.npy',
                      allow_pickle=True).tolist()

    # Initialize cross-validation splits
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rt_bins = pd.cut(RT, bins=10, labels=False)  # Stratify based on RT distribution
    indices = list(range(len(dataset)))

    # Initialize metrics storage for cross-validation
    fold_mae = []
    fold_r2 = []
    fold_rmse = []
    best_val_loss = float('inf')
    best_mae = float('inf')
    best_fold = 0
    fold_models = []

    # 5-fold cross-validation training loop
    for fold, (train_indices, val_indices) in enumerate(skf.split(indices, rt_bins)):
        print(f"\nTraining Fold {fold + 1}/5")

        # Construct graph dataset
        graph = Construct_dataset(dataset, RT, temperature_program, descriptor, column=TRANSFER_TARGET)

        # Split graph dataset into training and validation sets
        train_graph = [graph[i] for i in train_indices]
        val_graph = [graph[i] for i in val_indices]

        # Create PyTorch data loaders
        train_loader = DataLoader(train_graph, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_graph, batch_size=args.batch_size, shuffle=False)

        # Initialize GAT model with graph features
        node_features = graph[0].x.size(1)
        edge_features = graph[0].edge_attr.size(1)
        model = GAT(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            heads=args.heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

        # Set device (GPU/CPU) and move model to device
        if torch.cuda.is_available() and args.devices:
            args.device = torch.device(f"cuda:{args.devices[0]}")
        else:
            args.device = torch.device("cpu")
        model = model.to(args.device)

        # Initialize training components
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.SmoothL1Loss(beta=0.5)  # Robust loss function
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=100,
            min_lr=1e-5
        )

        # Initialize logging and early stopping
        writer = SummaryWriter(log_dir=f'{args.save_dir}/fold_{fold + 1}')
        early_stopping = EarlyStopping(
            patience=100,
            delta=0.001,
            path=f'saves/model_{TRANSFER_TARGET}/GAT_best_model_fold_{fold + 1}.pth'
        )

        # Main training loop for current fold
        for epoch in tqdm(range(args.epochs)):
            # Training step
            train_loss = train(
                model, args.device, train_loader, optimizer, criterion,
                args.l1_lambda, args.l2_lambda
            )

            # Validation step
            val_loss, val_mse, val_mae, val_r2 = eval(model, args.device, val_loader, criterion)

            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('MAE/val', val_mae, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'saves/model_{TRANSFER_TARGET}/GAT_best_model_fold_{fold + 1}.pth')

            # Apply early stopping after minimum epochs
            if epoch >= 200:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered in fold {fold + 1}!")
                    break

            # Save periodic checkpoints
            if (epoch + 1) % 100 == 0:
                torch.save(model.state_dict(),
                           f'saves/model_{TRANSFER_TARGET}/model_save_{epoch + 1}_fold_{fold + 1}.pth')

            # Print and log training progress
            print(
                f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, LR: {current_lr:.6f}')
            train_process = f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, LR: {current_lr:.6f}'
            file_path = f'Output/GAT_model/{TRANSFER_TARGET}/{TRANSFER_TARGET}_train_process_fold_{fold + 1}.txt'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'a') as file:
                file.write(train_process + '\n')

        writer.close()

        # Load best model and evaluate on validation set
        model.load_state_dict(torch.load(f'saves/model_{TRANSFER_TARGET}/GAT_best_model_fold_{fold + 1}.pth',
                                         weights_only=True))
        y_true, y_pred, mae, r2, rmse = test(model, args.device, val_loader, fold)

        # Store fold metrics
        fold_mae.append(mae)
        fold_r2.append(r2)
        fold_rmse.append(rmse)
        fold_models.append(model)

        # Save fold predictions
        results_df = pd.DataFrame({
            'Actual RT': y_true.flatten(),
            'Predicted RT': y_pred.flatten()
        })
        results_df.to_csv(
            f'Output/GAT_model/{TRANSFER_TARGET}/{TRANSFER_TARGET}_predictions_fold_{fold + 1}.csv',
            index=False
        )
        print(f"\nFold {fold + 1} Evaluation metrics:")
        print(f"MAE: {mae:.4f}, R2: {r2:.4f}, RMSE: {rmse:.4f}")

        # Select best model for external validation based on MAE
        if fold == 0 or mae < best_mae:
            best_mae = mae
            best_fold = fold + 1
            # Copy best model to unified location
            shutil.copyfile(
                f'saves/model_{TRANSFER_TARGET}/GAT_best_model_fold_{fold + 1}.pth',
                f'saves/model_{TRANSFER_TARGET}/best_model.pth'
            )

    # Calculate cross-validation statistics
    avg_mae = np.mean(fold_mae)
    avg_r2 = np.mean(fold_r2)
    avg_rmse = np.mean(fold_rmse)
    std_mae = np.std(fold_mae)
    std_r2 = np.std(fold_r2)
    std_rmse = np.std(fold_rmse)

    # Print cross-validation results
    print("\nCross-Validation Results (5 folds):")
    print(f"Average MAE: {avg_mae:.4f} (±{std_mae:.4f})")
    print(f"Average R2: {avg_r2:.4f} (±{std_r2:.4f})")
    print(f"Average RMSE: {avg_rmse:.4f} (±{std_rmse:.4f})")

    # Save cross-validation summary to file
    cv_summary = (
        f"Cross-Validation Results (5 folds):\n"
        f"Average MAE: {avg_mae:.4f} (±{std_mae:.4f})\n"
        f"Average R2: {avg_r2:.4f} (±{std_r2:.4f})\n"
        f"Average RMSE: {avg_rmse:.4f} (±{std_rmse:.4f})"
    )
    with open(f'Output/GAT_model/{TRANSFER_TARGET}/{TRANSFER_TARGET}_cv_summary.txt', 'w') as file:
        file.write(cv_summary)

    print("\nPerforming ensemble prediction...")

    # Create DataLoader for entire dataset
    full_loader = DataLoader(graph, batch_size=args.batch_size, shuffle=False)

    # Perform ensemble prediction using all fold models
    y_true, y_pred_ensemble, ensemble_results = ensemble_predict(
        fold_models, args.device, full_loader
    )

    # Print ensemble results
    print("\nEnsemble Model Performance:")
    print(f"MAE: {ensemble_results.loc[1, 'Value']:.4f}")
    print(f"R2: {ensemble_results.loc[2, 'Value']:.4f}")
    print(f"RMSE: {ensemble_results.loc[0, 'Value']:.4f}")

    # Save ensemble prediction results
    ensemble_results_df = pd.DataFrame({
        'Actual RT': y_true.flatten(),
        'Predicted RT': y_pred_ensemble.flatten()
    })
    ensemble_results_df.to_csv(
        f'Output/GAT_model/{TRANSFER_TARGET}/{TRANSFER_TARGET}_ensemble_predictions.csv',
        index=False
    )

    # Save ensemble evaluation metrics
    ensemble_results.to_csv(
        f'Output/GAT_model/{TRANSFER_TARGET}/{TRANSFER_TARGET}_ensemble_metrics.csv',
        index=False
    )

    # Generate ensemble results plot
    plot(y_true, y_pred_ensemble,
         ensemble_results.loc[1, 'Value'],
         ensemble_results.loc[2, 'Value'],
         ensemble_results.loc[0, 'Value'],
         TRANSFER_TARGET, fold=-1)

elif DESK == 'Validate_External':
    """
    External Validation Mode:
    - Loads external dataset and processes molecular structures
    - Creates graph representations for external data
    - Loads best trained model and performs predictions
    - Saves validation results and prediction errors
    """
    # Load external validation dataset
    external_dataset = pd.read_csv('dataset/external.csv')
    dataset_target = external_dataset[external_dataset['Column'] == f'{TRANSFER_TARGET}']

    # Create directories for external validation
    os.makedirs(f"Output/Validate_External/external_data/{TRANSFER_TARGET}", exist_ok=True)
    os.makedirs(f"Output/Validate_External/results/{TRANSFER_TARGET}", exist_ok=True)

    # Save filtered external dataset
    dataset_target.to_csv(f"Output/Validate_External/external_data/{TRANSFER_TARGET}/{TRANSFER_TARGET}_orderly.csv")

    # Extract molecular information from external dataset
    Chiral_molecules_smiles = dataset_target['Chiral_molecules_smile'].values
    Column_molecules_smiles = [get_column_molecules_smiles(TRANSFER_TARGET)]

    # Process temperature program data
    temperature_program = external_dataset.iloc[:, 4:18]
    np.save(f'Output/Validate_External/external_data/{TRANSFER_TARGET}/external_temperature_program.npy',
            temperature_program)

    # Generate 3D molecular structures for external data
    error_smile = save_3D_mol(Chiral_molecules_smiles,
                              f'Output/Validate_External/external_data/{TRANSFER_TARGET}/Chiral_3D_mol')
    Column_molecules = save_3D_mol(Column_molecules_smiles, f'Output/Validate_External/external_data/{TRANSFER_TARGET}')

    print(f'error_{TRANSFER_TARGET}:', error_smile)

    # Save error information
    np.save(f'Output/Validate_External/external_data/{TRANSFER_TARGET}/error_{TRANSFER_TARGET}.npy',
            np.array(error_smile))

    # Save external dataset with descriptors
    save_dataset(
        Chiral_molecules_smiles,
        f'Output/Validate_External/external_data/{TRANSFER_TARGET}',
        f'Output/Validate_External/external_data/{TRANSFER_TARGET}/external_dataset',
        f'Output/Validate_External/external_data/{TRANSFER_TARGET}/external_dataset_descriptors',
        error_smile,
        TRANSFER_TARGET
    )

    # Load and process external graph data
    external_graph = load_external_data()

    # Set device and initialize model architecture
    device = args.device
    node_features = external_graph[0].x.size(1)
    edge_features = external_graph[0].edge_attr.size(1)

    # Initialize GAT model with same architecture as training
    model = GAT(node_features=node_features, edge_features=edge_features,
                hidden_dim=args.hidden_dim, output_dim=args.output_dim,
                heads=args.heads, num_layers=args.num_layers,
                dropout=args.dropout)

    # Load the best trained model weights
    model.load_state_dict(torch.load(f'saves/model_{TRANSFER_TARGET}/best_model.pth',
                                     weights_only=True))

    # Set device and move model
    if torch.cuda.is_available() and args.devices:
        args.device = torch.device(f"cuda:{args.devices[0]}")
    else:
        args.device = torch.device("cpu")
    model = model.to(args.device)

    # Create data loader for external validation
    external_loader = DataLoader(external_graph, batch_size=args.batch_size, shuffle=False)

    # Perform external validation
    y_true, y_pred = Validate_External(model, args.device, external_loader)

    # Save external validation results with error analysis
    external_results_df = pd.DataFrame({
        'Actual RT': y_true,
        'Predicted RT': y_pred,
        'Error': abs(y_true - y_pred)
    })
    external_results_df.to_csv(
        f'Output/Validate_External/results/{TRANSFER_TARGET}/{TRANSFER_TARGET}_external_predictions.csv', index=False)
    print(f"Results saved to {TRANSFER_TARGET}_external_predictions.csv")

else:
    """
    Invalid Configuration:
    - Display error message for invalid DESK values
    """
    print("Invalid DESK value. Please choose 'Dataset_construct', 'Train', 'Validate_External'")