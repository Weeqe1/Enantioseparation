# Standard library imports
import json
import os
import random
import shutil
import tempfile
import warnings
from typing import Dict, Tuple, List

# Third-party library imports
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from bayes_opt import BayesianOptimization
from torch_geometric.data import DataLoader, Data
from tqdm import tqdm

# Local module imports
from config import TRANSFER_TARGET, RANDOM_SEED
from Feature_calculation import Construct_dataset
from GAT_model import GAT
from train import train, eval, augment_data

# Configure warnings and multiprocessing
warnings.filterwarnings("ignore", category=UserWarning)
mp.set_start_method('spawn', force=True)

class CheckpointManager:
    """
    Manages checkpoint saving and loading for Bayesian optimization process.
    Keeps track of tried parameter combinations and their scores to avoid redundant evaluations.
    """

    def __init__(self, checkpoint_dir: str = 'Output/Parameter_tuning_bayes',
                 transfer_target: str = TRANSFER_TARGET):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
            transfer_target: Target column name for transfer learning
        """
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f'{transfer_target}_optimization_checkpoint.json')
        self.tried_params_cache: Dict[Tuple, float] = {}
        self.best_score = float('-inf')

        # Remove existing checkpoint file to start fresh
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        self.load_checkpoint()

    def load_checkpoint(self):
        """
        Load previously saved optimization checkpoint from file.

        Returns:
            Tuple of (tried_params_cache, best_score)
        """
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                    self.tried_params_cache = {tuple(sorted(d['params'].items())): d['score'] for d in
                                               data['tried_params']}
                    self.best_score = data['best_score']
            except (IOError, json.JSONDecodeError) as e:
                print(f"Failed to load checkpoint: {e}. Starting with empty cache.")
        return self.tried_params_cache, self.best_score

    def save_checkpoint(self, params: Dict, score: float):
        """
        Save current optimization progress to checkpoint file.

        Args:
            params: Parameter dictionary to save
            score: Associated score for the parameters
        """
        params_tuple = tuple(sorted(params.items()))
        self.tried_params_cache[params_tuple] = score

        # Update best score if current score is better and valid
        if score > self.best_score and not (np.isnan(score) or np.isinf(score)):
            self.best_score = score

        def convert_to_serializable(obj):
            """Convert numpy types to JSON-serializable types."""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        # Prepare checkpoint data for JSON serialization
        checkpoint_data = {
            'tried_params': [{'params': convert_to_serializable(dict(k)), 'score': convert_to_serializable(v)}
                             for k, v in self.tried_params_cache.items()],
            'best_score': convert_to_serializable(self.best_score)
        }

        # Safely write to file using temporary file to avoid corruption
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile('w', delete=False, dir=self.checkpoint_dir, suffix='.json') as temp:
                temp_file = temp.name
                json.dump(checkpoint_data, temp, indent=4)
            shutil.move(temp_file, self.checkpoint_path)
        except IOError as e:
            print(f"Failed to save checkpoint: {e}")
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    def is_similar_params_tried(self, params: Dict) -> bool:
        """
        Check if similar parameters have been tried before.

        Args:
            params: Parameter dictionary to check

        Returns:
            True if similar parameters have been tried, False otherwise
        """
        params_tuple = tuple(sorted(params.items()))
        return params_tuple in self.tried_params_cache

class ModelEvaluator:
    """
    Handles model evaluation for Bayesian optimization.
    Manages parameter processing, model training, and performance evaluation.
    """

    # Parameter value ranges for discrete optimization
    param_values = {
        'lr': [1e-5, 1e-4, 1e-3, 1e-2],
        'batch_size': [8, 16, 32],
        'hidden_dim': [32, 48, 64, 128],
        'output_dim': [8, 12, 16, 32],
        'heads': [2, 4, 6, 8, 10, 12, 14, 16],
        'num_layers': [2, 4, 6, 8, 10, 12, 14, 16],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'l1_lambda': [1e-5, 1e-4, 1e-3, 1e-2],
        'l2_lambda': [1e-5, 1e-4, 1e-3, 1e-2],
        'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2]
    }

    def __init__(self, train_dataset, test_dataset, node_features, edge_features,
                 checkpoint_dir='Output/Parameter_tuning_bayes'):
        """
        Initialize model evaluator.

        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            node_features: Number of node features
            edge_features: Number of edge features
            checkpoint_dir: Directory for checkpoints
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.node_features = node_features
        self.edge_features = edge_features
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.best_score = float('-inf')
        self.failed_attempts = 0
        self.max_failed_attempts = 10
        self.best_model = None

    def evaluate_model(self, **params) -> float:
        """
        Evaluate model with given parameters.

        Args:
            **params: Model hyperparameters

        Returns:
            Model performance score (negative MSE)
        """
        try:
            # Process parameters to actual values
            processed_params = self.process_params(params)

            # Skip if similar parameters have been tried
            if self.checkpoint_manager.is_similar_params_tried(processed_params):
                return self.get_fallback_score()

            # Create data loaders
            train_loader = DataLoader(self.train_dataset, batch_size=processed_params['batch_size'],
                                      shuffle=True, pin_memory=True, num_workers=0, prefetch_factor=None)
            test_loader = DataLoader(self.test_dataset, batch_size=processed_params['batch_size'],
                                     shuffle=False, pin_memory=True, num_workers=0, prefetch_factor=None)

            # Initialize model
            model = GAT(self.node_features, self.edge_features, processed_params['hidden_dim'],
                        processed_params['output_dim'], processed_params['heads'],
                        processed_params['num_layers'], processed_params['dropout']).to(self.device)

            # Initialize optimizer and loss function
            optimizer = optim.AdamW(model.parameters(), lr=processed_params['lr'],
                                    weight_decay=processed_params['weight_decay'])
            criterion = torch.nn.SmoothL1Loss(beta=0.5)

            # Initialize learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.8,
                patience=100,
                min_lr=1e-5
            )

            # Training loop setup
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 100

            # Training loop with progress bar
            pbar = tqdm(range(50000), desc="Training Epochs", leave=False)
            for epoch in pbar:
                # Data augmentation during training
                for data in train_loader:
                    data = data.to(self.device)
                    if model.training:
                        data = augment_data(data)

                # Train for one epoch
                train_loss = train(model, self.device, train_loader, optimizer, criterion,
                                   processed_params['l1_lambda'], processed_params['l2_lambda'])

                # Evaluate on validation set
                val_loss, val_mse, val_mae, val_r2 = eval(model, self.device, test_loader, criterion)
                scheduler.step(val_loss)

                # Early stopping logic (starts after epoch 200)
                if epoch >= 200:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model temporarily
                        torch.save(model.state_dict(),
                                   f'Output/Parameter_tuning_bayes/best_model_temp_{TRANSFER_TARGET}.pth')
                    else:
                        patience_counter += 1

                    # Stop if patience exceeded
                    if patience_counter >= max_patience:
                        # Load best model
                        model.load_state_dict(
                            torch.load(f'Output/Parameter_tuning_bayes/best_model_temp_{TRANSFER_TARGET}.pth',
                                       weights_only=True))
                        break

                # Update progress bar
                pbar.set_postfix({
                    'epoch': epoch,
                    'train_loss': f'{train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}',
                    'val_mse': f'{val_mse:.6f}',
                    'val_r2': f'{val_r2:.6f}',
                    'patience_counter': f'{patience_counter}'
                })
            pbar.close()

            # Final evaluation on test set
            with torch.no_grad():
                _, test_mse, _, _ = eval(model, self.device, test_loader, criterion)

            # Handle invalid scores
            if np.isnan(test_mse) or np.isinf(test_mse):
                self.failed_attempts += 1
                return self.get_fallback_score()

            # Convert MSE to score (negative for maximization)
            score = -test_mse
            if np.isnan(score) or np.isinf(score):
                score = -1e10

            self.failed_attempts = 0
            self.checkpoint_manager.save_checkpoint(processed_params, score)

            # Update best model if current is better
            if self.best_model is None or score > self.best_score:
                self.best_model = model

            return score

        except Exception as e:
            print(f"Error during evaluation: {e}")
            self.failed_attempts += 1
            return self.get_fallback_score()

    def get_fallback_score(self) -> float:
        """
        Get fallback score when evaluation fails or parameters already tried.

        Returns:
            Fallback score based on best known score
        """
        tried_params_cache, best_score = self.checkpoint_manager.load_checkpoint()
        if best_score > float('-inf'):
            return best_score - 100
        return -1e10

    @staticmethod
    def process_params(params):
        """
        Convert continuous parameter values to discrete choices.

        Args:
            params: Dictionary of continuous parameters

        Returns:
            Dictionary of processed discrete parameters
        """
        processed = params.copy()

        # Map continuous values to discrete choices
        for param, values in ModelEvaluator.param_values.items():
            index = int(round(params[param]))
            index = min(max(index, 0), len(values) - 1)
            processed[param] = values[index]

        # Ensure integer parameters are integers
        int_params = ['batch_size', 'hidden_dim', 'output_dim', 'heads', 'num_layers']
        for param in processed:
            if param in int_params:
                processed[param] = int(processed[param])
            elif isinstance(processed[param], np.floating):
                processed[param] = float(processed[param])
        return processed

    def get_best_model(self):
        """
        Get the best model found during optimization.

        Returns:
            Best trained model
        """
        return self.best_model


class CustomBayesianOptimization(BayesianOptimization):
    """
    Custom Bayesian optimization class with enhanced parameter display.
    Extends the base BayesianOptimization to show processed parameters during optimization.
    """

    def __init__(self, f, pbounds, process_params_fn, random_state=None, verbose=0):
        """
        Initialize custom Bayesian optimization.

        Args:
            f: Objective function to maximize
            pbounds: Parameter bounds
            process_params_fn: Function to process parameters for display
            random_state: Random state for reproducibility
            verbose: Verbosity level
        """
        super().__init__(f=f, pbounds=pbounds, random_state=random_state, verbose=verbose)
        self.process_params_fn = process_params_fn

    def _print_iteration(self, iteration, params, value):
        """
        Print optimization iteration with processed parameters.

        Args:
            iteration: Current iteration number
            params: Raw parameter values
            value: Objective function value
        """
        processed_params = self.process_params_fn(params)  # Call static method directly
        param_str = " | ".join([f"{k}: {v:.7f}" if isinstance(v, float) else f"{k}: {v}"
                                for k, v in processed_params.items()])
        print(f"| {iteration:2d} | {value:.6f} | {param_str} |")


def bayesian_optimization(train_dataset, test_dataset, node_features, edge_features) -> Dict:
    """
    Perform Bayesian optimization to find best hyperparameters.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        node_features: Number of node features
        edge_features: Number of edge features

    Returns:
        Dictionary of best parameters found
    """
    # Define parameter bounds for Bayesian optimization (indices for discrete choices)
    pbounds = {
        'lr': (0, len(ModelEvaluator.param_values['lr']) - 1),
        'batch_size': (0, len(ModelEvaluator.param_values['batch_size']) - 1),
        'hidden_dim': (0, len(ModelEvaluator.param_values['hidden_dim']) - 1),
        'output_dim': (0, len(ModelEvaluator.param_values['output_dim']) - 1),
        'heads': (0, len(ModelEvaluator.param_values['heads']) - 1),
        'num_layers': (0, len(ModelEvaluator.param_values['num_layers']) - 1),
        'dropout': (0, len(ModelEvaluator.param_values['dropout']) - 1),
        'l1_lambda': (0, len(ModelEvaluator.param_values['l1_lambda']) - 1),
        'l2_lambda': (0, len(ModelEvaluator.param_values['l2_lambda']) - 1),
        'weight_decay': (0, len(ModelEvaluator.param_values['weight_decay']) - 1)
    }

    # Initialize evaluator and optimizer
    evaluator = ModelEvaluator(train_dataset, test_dataset, node_features, edge_features)
    optimizer = CustomBayesianOptimization(
        f=evaluator.evaluate_model,
        pbounds=pbounds,
        process_params_fn=ModelEvaluator.process_params,
        random_state=RANDOM_SEED,
        verbose=2
    )
    optimizer.set_gp_params(alpha=1e-6, n_restarts_optimizer=10)

    try:
        # Run Bayesian optimization
        optimizer.maximize(init_points=20, n_iter=200)

        # Handle case where no solution was found
        if optimizer.max is None:
            print("Optimization did not find any valid solution")
            _, best_score = evaluator.checkpoint_manager.load_checkpoint()
            if best_score > float('-inf'):
                # Find parameters with best score from cache
                best_trial_params = None
                for params_tuple, score in evaluator.checkpoint_manager.tried_params_cache.items():
                    if score == best_score:
                        best_trial_params = dict(params_tuple)
                        break
                if best_trial_params:
                    return best_trial_params
            raise ValueError("No valid parameters found")

        return evaluator.process_params(optimizer.max['params'])

    except Exception as e:
        print(f"Optimization error: {e}")
        # Try to recover best parameters from checkpoint
        _, best_score = evaluator.checkpoint_manager.load_checkpoint()
        if best_score > float('-inf'):
            best_trial_params = None
            for params_tuple, score in evaluator.checkpoint_manager.tried_params_cache.items():
                if score == best_score:
                    best_trial_params = dict(params_tuple)
                    break
            if best_trial_params:
                return best_trial_params
        raise ValueError("No valid parameters found")


def load_data(transfer_target: str) -> Tuple[List[Data], np.ndarray, np.ndarray]:
    """
    Load and preprocess data for the specified transfer target.

    Args:
        transfer_target: Target column name for transfer learning

    Returns:
        Tuple of (graph_data, retention_times, temperature_program)
    """
    # Load gas chromatography data
    GC = pd.read_csv(f'Output/GAT_model/{transfer_target}/dataset/{transfer_target}_orderly.csv')

    # Remove bad samples
    bad_index = np.load(f'Output/GAT_model/{transfer_target}/dataset/error_{transfer_target}.npy')
    GC = GC.drop(bad_index).reset_index(drop=True)

    # Load molecular descriptors and other data
    descriptor = np.load(
        f'Output/GAT_model/{transfer_target}/dataset/dataset_{transfer_target}_orderly_descriptors.npy')
    RT = GC['RT'].values
    temperature_program = np.load(
        f'Output/GAT_model/{transfer_target}/dataset/{transfer_target}_temperature_program.npy')
    dataset = np.load(f'Output/GAT_model/{transfer_target}/dataset/dataset_{transfer_target}_orderly.npy',
                      allow_pickle=True).tolist()

    # Construct graph dataset
    graph = Construct_dataset(dataset, RT, temperature_program, descriptor, column=transfer_target)
    return graph, RT, temperature_program


def split_dataset(graph: List[Data], random_seed: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Split dataset into training and test sets.

    Args:
        graph: List of graph data objects
        random_seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    total_num = len(graph)
    train_size = int(0.8 * total_num)
    test_size = total_num - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        graph, [train_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    return train_dataset, test_dataset


def main():
    """
    Main function to run Bayesian hyperparameter optimization.
    """
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print(f"{TRANSFER_TARGET} Start calculation")

    # Create output directory
    output_dir = "Output/Parameter_tuning_bayes"
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    graph, _, _ = load_data(TRANSFER_TARGET)
    train_dataset, test_dataset = split_dataset(graph, RANDOM_SEED)

    # Get feature dimensions
    node_features = graph[0].x.size(1)
    edge_features = graph[0].edge_attr.size(1)

    try:
        # Run Bayesian optimization
        best_params = bayesian_optimization(train_dataset, test_dataset, node_features, edge_features)
        print(f"{TRANSFER_TARGET}_Best Parameters: {best_params}")

        # Save results to file
        with open(os.path.join(output_dir, 'results.txt'), 'a') as file:
            final_process = f"{TRANSFER_TARGET}_Best Parameters: {best_params}"
            file.write(final_process + '\n')

    except Exception as e:
        print(f"Error in optimization process: {e}")
        print("Check checkpoint file for best parameters found so far.")


if __name__ == '__main__':
    main()