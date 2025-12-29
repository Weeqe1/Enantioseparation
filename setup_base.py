"""
Main setup script for baseline model training and validation.

This script provides three main functionalities based on the DESK configuration:
1. Dataset construction and processing
2. Model training and evaluation
3. External dataset validation
"""

# Standard library imports
import warnings

# Third-party imports
import pandas as pd

# Local imports
from Baseline_model import (
    create_output_directories,
    load_dataset,
    prepare_data,
    process_dataset,
    train_and_evaluate_models,
    validate_external_dataset
)
from config import DESK, MODEL_NAME, TRANSFER_TARGET

# Suppress user warnings to keep output clean
warnings.filterwarnings("ignore", category=UserWarning)

# Main execution logic based on DESK configuration
if DESK == 'Dataset_construct':
    """
    Dataset Construction Mode:
    - Creates necessary output directories
    - Processes and prepares the dataset for training
    """
    # Create output directories for storing results
    create_output_directories()

    # Process and construct the dataset
    process_dataset()

elif DESK == 'Train':
    """
    Training Mode:
    - Loads the preprocessed dataset
    - Prepares training/validation/test splits
    - Trains multiple models and evaluates their performance
    - Saves evaluation results to CSV file
    """
    # Load the complete dataset with all components
    gc, descriptor, all_smile, rt, dataset, temperature_program = load_dataset()

    # Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(dataset, rt, temperature_program, descriptor)

    # Train multiple models and collect evaluation results
    results = train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)

    # Convert results to DataFrame and display summary
    results_summary_df = pd.DataFrame(results)
    print(results_summary_df)

    # Save evaluation results to CSV file
    results_summary_df.to_csv(f'Output/Baseline/{TRANSFER_TARGET}/models_evaluation.csv', index=False)

elif DESK == 'Validate_External':
    """
    External Validation Mode:
    - Validates the trained model on external dataset
    - Uses the specified model for validation
    """
    # Validate model performance on external dataset
    validate_external_dataset('dataset/external_data.csv', MODEL_NAME)

else:
    """
    Invalid Configuration:
    - Display error message for invalid DESK values
    """
    print("Invalid DESK value. Please choose 'Dataset_construct', 'Train', or 'Validate_External'.")