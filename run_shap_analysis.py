"""
SHAP Analysis for Chromatographic Retention Time Prediction

This script performs comprehensive SHAP (SHapley Additive exPlanations) analysis
for predicting retention times in chromatographic separations using molecular
descriptors and chiral features.

Key Features:
- Chiral feature engineering from SMILES strings
- Mordred molecular descriptor calculation
- XGBoost model training with cross-column analysis
- SHAP-based feature importance analysis
- Multi-criteria feature selection
- Comprehensive visualization including circular plots
"""

# Standard Library Imports
import logging
import multiprocessing
import os
import warnings
from contextlib import contextmanager

# Third-Party Scientific Computing
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colorbar import ColorbarBase

# Machine Learning and Model Interpretation
import shap
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Chemistry and Molecular Informatics
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem

# Configuration and Setup
# Suppress specific warnings from XGBoost to maintain cleaner output during execution
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging for comprehensive execution tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Performance Configuration
# Determine the number of CPU cores for parallel processing to optimize performance
NUM_CORES = 1

# GPU Availability Check
# Verify GPU availability for accelerating XGBoost computations
try:
    import cupy

    GPU_AVAILABLE = True
    logger.info("GPU detected. Utilizing GPU acceleration for XGBoost operations.")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("No GPU detected (cupy not installed). Executing on CPU.")

# Chromatographic Column Data
# Define chromatographic column SMILES dictionary for descriptor calculation
column_molecules_dict = {
    'Cyclosil_B': 'CC(C)(C)[Si](C)(C)OC[C@@H]1C([O])[C@@H](OC)[C@@H](OC)[C@@H]([O])O1',
    'Cyclodex_B': 'OC[C@@H]1C([C@H]([C@H]([C@H](O1)[O])O)O)[O]',
    'HP_chiral_20β': 'OC[C@@H]1C([C@H]([C@H]([C@H](O1)[O])O)O)[O]',
    'CP_Cyclodextrin_β_2,3,6_M_19': 'OC[C@@H]1C([C@H]([C@H]([C@H](O1)[O])O)O)[O]',
    'CP_Chirasil_D_Val': 'CC(C)[C@H](NC(=O)C(C)(C)C)C(=O)O',
    'CP_Chirasil_Dex_CB': 'OC[C@@H]1C([C@H]([C@H]([C@H](O1)[O])O)O)[O]',
    'CP_Chirasil_L_Val': 'CC(C)[C@@H](NC(=O)C(C)(C)C)C(=O)O'
}


@contextmanager
def plot_context(figsize=(10, 6), **kwargs):
    """
    Context manager for creating and managing matplotlib figures.

    This context manager ensures that figures are properly closed after use 
    to prevent memory leaks during extensive plotting operations.

    Args:
        figsize (tuple): Dimensions of the figure (width, height) in inches
        **kwargs: Additional arguments to pass to plt.figure()

    Yields:
        matplotlib.figure.Figure: The created figure object
    """
    fig = plt.figure(figsize=figsize, **kwargs)
    yield fig
    plt.close(fig)


def load_experimental_data(file_path='dataset/data.csv'):
    """
    Load experimental chromatographic data from CSV file.

    Args:
        file_path (str): The relative or absolute path to the data file

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded experimental data

    Raises:
        FileNotFoundError: If the specified data file does not exist
        Exception: For other errors encountered during data loading
    """
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at: {file_path}")
        raise FileNotFoundError(f"Data file {file_path} not found")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error encountered while loading data: {e}")
        raise


def characterize_chiral_features(smiles_string, index):
    """
    Analyze SMILES string to extract chiral stereochemical information.

    This function identifies and counts chiral centers, determines R/S configurations,
    and classifies the overall chirality type of the molecule.

    Args:
        smiles_string (str): The SMILES notation of a molecule
        index (int): The original index of the molecule in the dataset (for logging)

    Returns:
        tuple: A tuple containing:
            - total_chiral_centers (int): Total number of chiral centers
            - r_count (int): Number of R-configured chiral centers  
            - s_count (int): Number of S-configured chiral centers
            - chiral_type (int): Chirality classification (0=achiral, 1=all R, 2=all S, 3=mixed)
        Returns (0, 0, 0, 0) if the SMILES is invalid or an error occurs
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            logger.warning(f"Invalid SMILES string encountered at index {index}: '{smiles_string}'")
            return 0, 0, 0, 0

        # Find all chiral centers in the molecule
        chiral_centers = list(AllChem.FindMolChiralCenters(mol, includeUnassigned=True))

        # Count R and S configurations
        r_count = sum(1 for _, tag in chiral_centers if tag == 'R')
        s_count = sum(1 for _, tag in chiral_centers if tag == 'S')
        total_chiral = len(chiral_centers)

        # Classify chirality type
        chiral_type = 0  # Default to achiral or mixed
        if total_chiral > 0:
            if r_count == total_chiral:
                chiral_type = 1  # All R configuration
            elif s_count == total_chiral:
                chiral_type = 2  # All S configuration
            else:
                chiral_type = 3  # Mixed R/S configuration

        return total_chiral, r_count, s_count, chiral_type

    except Exception as e:
        logger.warning(f"Error during chiral feature characterization for SMILES at index {index}: {e}")
        return 0, 0, 0, 0


def compute_molecular_descriptors(molecule, index, descriptor_calculator, prefix=""):
    """
    Calculate Mordred molecular descriptors for a given RDKit molecule.

    This function computes a comprehensive set of molecular descriptors using
    the Mordred descriptor calculator, which includes 2D and 3D molecular features.

    Args:
        molecule (rdkit.Chem.Mol): The RDKit molecule object
        index (int): The original index of the molecule (for logging purposes)
        descriptor_calculator (mordred.Calculator): An initialized Mordred Calculator instance
        prefix (str): Prefix to add to descriptor names for identification

    Returns:
        pd.Series: A pandas Series containing computed descriptors with prefixes,
                   or None if computation fails
    """
    try:
        if molecule is None:
            logger.warning(f"Invalid molecule object provided at index {index}. Skipping descriptor computation.")
            return None

        # Calculate all descriptors for the molecule
        descriptors_obj = descriptor_calculator(molecule)
        descriptor_series = pd.Series(descriptors_obj.asdict())

        # Add prefix to descriptor names for feature identification
        descriptor_series.index = [f"{prefix}{str(desc)}" for desc in descriptor_series.index]

        return descriptor_series

    except Exception as e:
        logger.warning(f"Error computing Mordred descriptors for molecule at index {index}: {e}")
        return None


def determine_elbow_point(importances_array):
    """
    Identify the 'elbow point' in feature importance distribution.

    This function uses the largest distance from the line connecting the first 
    and last points to find the optimal cutoff for feature selection. The elbow
    point represents where additional features provide diminishing returns.

    Args:
        importances_array (np.ndarray): A sorted (descending) array of feature importances

    Returns:
        tuple: A tuple containing:
            - elbow_index (int): The index of the elbow point
            - elbow_importance (float): The importance value at the elbow point
        Returns (0, 0) or (0, first_importance) for arrays with fewer than 3 elements
    """
    n_features = len(importances_array)
    if n_features < 3:
        return 0, importances_array[0] if n_features > 0 else 0

    # Calculate cumulative importance and normalize
    cumulative_importance = np.cumsum(importances_array)
    normalized_cumulative = cumulative_importance / cumulative_importance[-1]

    # Calculate distances from each point to the diagonal line (perfect uniform distribution)
    distances = []
    for i in range(n_features):
        # Point coordinates: (x, y) where x is normalized index, y is normalized cumulative importance
        x = i / (n_features - 1)
        y = normalized_cumulative[i]

        # Distance from point (x0, y0) to line y=x is |x - y| / sqrt(2)
        dist = np.abs(x - y) / np.sqrt(2)
        distances.append(dist)

    # Find the point with maximum distance (elbow point)
    elbow_idx = np.argmax(distances)
    return elbow_idx, importances_array[elbow_idx]


def apply_multicriteria_feature_selection(shap_importances_df, min_absolute_importance=0.05, domain_keywords=None):
    """
    Select important features using multiple selection criteria.

    This function implements a comprehensive feature selection strategy combining:
    1. Elbow point method on SHAP importance distribution
    2. Absolute importance threshold filtering
    3. Domain-specific keyword matching (e.g., 'Chiral' features)
    4. Minimum feature count guarantee

    Args:
        shap_importances_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
        min_absolute_importance (float): Minimum absolute SHAP importance threshold
        domain_keywords (list): List of keywords to identify domain-specific features

    Returns:
        pd.DataFrame: A DataFrame containing selected features with selection criteria flags,
                      sorted by importance and including importance rankings
    """
    if domain_keywords is None:
        domain_keywords = ['Chiral', 'chiral']

    df = shap_importances_df.copy()

    # Apply elbow point criterion
    sorted_importances = df['importance'].sort_values(ascending=False).values
    elbow_idx, elbow_threshold = determine_elbow_point(sorted_importances)
    df['selected_by_elbow'] = df['importance'] >= elbow_threshold

    # Apply absolute importance threshold
    df['selected_by_absolute_threshold'] = df['importance'] >= min_absolute_importance

    # Apply domain knowledge criterion (keyword matching)
    domain_pattern = '|'.join(domain_keywords)
    df['selected_by_domain_knowledge'] = df['feature'].str.contains(domain_pattern, case=False, na=False)

    # Combine all selection criteria
    df['is_selected'] = df['selected_by_elbow'] | df['selected_by_absolute_threshold'] | df[
        'selected_by_domain_knowledge']

    # Ensure minimum number of features are selected (prevent too aggressive filtering)
    if df['is_selected'].sum() < 5:
        top_n_to_force = min(5, len(df))
        df.loc[df['importance'].nlargest(top_n_to_force).index, 'is_selected'] = True

    # Add importance ranking for analysis
    df['importance_rank'] = df['importance'].rank(ascending=False)
    selected_df = df[df['is_selected']].sort_values('importance', ascending=False)

    logger.info(f"Selected {len(selected_df)} features after multi-criteria selection")
    return selected_df


def run_shap_analysis(
        title_fontsize=16, label_fontsize=14, tick_fontsize=12, legend_fontsize=10,
        shap_summary_feature_fontsize=10, heatmap_cbar_fontsize=10, heatmap_tick_fontsize=8
):
    """
    Main orchestration function for SHAP analysis pipeline.

    This function coordinates the entire analysis workflow:
    1. Data loading and preprocessing
    2. Chiral feature engineering 
    3. Molecular descriptor calculation
    4. Unified cross-column model training
    5. Column-wise SHAP analysis
    6. Cross-column comparison and visualization

    Args:
        title_fontsize (int): Font size for plot titles
        label_fontsize (int): Font size for axis labels  
        tick_fontsize (int): Font size for tick labels
        legend_fontsize (int): Font size for legends
        shap_summary_feature_fontsize (int): Font size for SHAP summary feature names
        heatmap_cbar_fontsize (int): Font size for heatmap colorbar
        heatmap_tick_fontsize (int): Font size for heatmap tick labels
    """

    # Step 1: Data Loading and Initial Setup
    logger.info("Starting SHAP analysis pipeline...")
    df = load_experimental_data()

    # Step 2: Chiral Feature Engineering
    logger.info("Initiating chiral feature engineering...")
    chiral_features = Parallel(n_jobs=NUM_CORES)(
        delayed(characterize_chiral_features)(smi, idx)
        for idx, smi in enumerate(df['Chiral_molecules_smile'])
    )
    chiral_features_df = pd.DataFrame(
        chiral_features,
        columns=['Total_Chiral', 'R_Count', 'S_Count', 'Chiral_Type']
    )
    df = pd.concat([df, chiral_features_df], axis=1)
    logger.info("Chiral feature engineering completed.")

    # Step 3: Molecular Descriptor Calculation for Chiral Molecules
    logger.info("Calculating Mordred molecular descriptors for chiral molecules...")
    mordred_calculator = Calculator(descriptors, ignore_3D=False)
    molecules = [Chem.MolFromSmiles(smi) for smi in df['Chiral_molecules_smile']]

    # Calculate descriptors with 'mol_' prefix for chiral molecules
    computed_mol_descriptors_list = Parallel(n_jobs=NUM_CORES)(
        delayed(compute_molecular_descriptors)(mol, idx, mordred_calculator, prefix="mol_")
        for idx, mol in enumerate(molecules)
    )

    # Process and clean molecular descriptors
    computed_mol_descriptors_list = [d for d in computed_mol_descriptors_list if d is not None]
    mol_descriptor_df = pd.DataFrame(computed_mol_descriptors_list)
    mol_descriptor_df = mol_descriptor_df.apply(pd.to_numeric, errors='coerce')
    mol_descriptor_df = mol_descriptor_df.select_dtypes(include=[np.number])
    mol_descriptor_df = mol_descriptor_df.fillna(mol_descriptor_df.mean())
    mol_descriptor_df = mol_descriptor_df.loc[:, mol_descriptor_df.nunique() > 1]  # Remove constant features

    logger.info(f"Mordred descriptor calculation for chiral molecules completed. "
                f"Generated {len(mol_descriptor_df.columns)} descriptors with 'mol_' prefix.")

    # Step 4: Molecular Descriptor Calculation for Column Molecules  
    logger.info("Calculating Mordred molecular descriptors for column molecules...")
    column_descriptors_data = {}

    for column_name, smiles_string in column_molecules_dict.items():
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            logger.warning(f"Invalid SMILES string for column '{column_name}': '{smiles_string}'. "
                           f"Skipping descriptor calculation.")
            column_descriptors_data[column_name] = None
            continue

        # Calculate descriptors with 'columo_' prefix for column molecules
        col_desc_series = compute_molecular_descriptors(mol, -1, mordred_calculator, prefix="columo_")
        if col_desc_series is not None:
            column_descriptors_data[column_name] = col_desc_series.apply(pd.to_numeric, errors='coerce')
        else:
            column_descriptors_data[column_name] = None

    # Create DataFrame from column descriptors
    col_desc_raw_df = pd.DataFrame.from_dict(column_descriptors_data, orient='index')
    col_desc_raw_df = col_desc_raw_df.apply(pd.to_numeric, errors='coerce')
    logger.info(f"Mordred descriptor calculation for column molecules completed.")

    # Step 5: Unified Cross-Column Modeling Strategy
    logger.info("Initiating cross-column unified modeling...")

    # Expand column descriptors to match the main dataset structure
    column_descriptors_expanded = []
    for column_name in df['Column']:
        if column_name in col_desc_raw_df.index and col_desc_raw_df.loc[column_name] is not None:
            col_desc = col_desc_raw_df.loc[column_name].copy()
            column_descriptors_expanded.append(col_desc)
        else:
            logger.warning(f"No valid descriptors found for column '{column_name}'. Using zeros.")
            # Create zero-filled series for missing column descriptors
            empty_desc = pd.Series(0, index=col_desc_raw_df.columns)
            column_descriptors_expanded.append(empty_desc)

    column_descriptors_expanded = pd.DataFrame(column_descriptors_expanded, index=df.index)

    # Combine all feature types into unified feature matrix
    X_all = pd.concat([
        mol_descriptor_df.reset_index(drop=True),  # Molecular descriptors
        df[['Total_Chiral', 'R_Count', 'S_Count', 'Chiral_Type']].reset_index(drop=True),  # Chiral features
        column_descriptors_expanded.reset_index(drop=True)  # Column descriptors
    ], axis=1)

    # Remove any remaining constant features
    X_all = X_all.loc[:, X_all.nunique() > 1]
    y_all = df['RT']  # Target variable (retention time)

    logger.info(f"Total features in unified dataset: {X_all.shape[1]}")
    logger.info(f"Column descriptor features added: {len(column_descriptors_expanded.columns)}")

    # Step 6: Data Preprocessing and Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_all),
        columns=X_all.columns,
        index=X_all.index
    )

    # Split data for model training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=42
    )
    logger.info(f"Unified data split: {len(X_train)} training samples, {len(X_test)} test samples.")

    # Step 7: XGBoost Model Training
    xgb_parameters = {
        'n_estimators': 5000,
        'random_state': 42,
        "learning_rate": 0.001,
        'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
        'device': 'cuda' if GPU_AVAILABLE else 'cpu',
        'early_stopping_rounds': 100,
        'eval_metric': 'rmse'
    }

    model = xgb.XGBRegressor(**xgb_parameters)
    logger.info(f"Training unified XGBoost model (device: {'GPU' if GPU_AVAILABLE else 'CPU'})...")

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    # Evaluate model performance
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    logger.info(f"Unified model performance: Training R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")

    # Initialize SHAP explainer for the unified model
    explainer = shap.TreeExplainer(model)

    # Step 8: Column-wise SHAP Analysis Using Unified Model
    unique_columns = df['Column'].unique()
    analysis_results = {}

    for column_name in unique_columns:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Commencing SHAP analysis for chromatographic column: '{column_name}'")
        logger.info(f"{'=' * 60}")

        # Create results directory for current column
        result_directory = f"Output/SHAP_Results/{column_name}"
        try:
            os.makedirs(result_directory, exist_ok=True)
            logger.info(f"Created results directory: {result_directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {result_directory}. Skipping analysis for this column: {e}")
            continue

        # Filter data specific to the current chromatographic column
        column_data_mask = df['Column'] == column_name
        X_column = X_scaled[column_data_mask]

        if len(X_column) < 5:
            logger.warning(f"Insufficient data samples ({len(X_column)}) for column '{column_name}'. "
                           f"Skipping analysis.")
            continue

        # Calculate SHAP values for this column's data subset
        shap_values_for_column = explainer.shap_values(X_column)

        # Aggregate SHAP importances (mean absolute SHAP value across samples)
        shap_importances_df = pd.DataFrame({
            'feature': X_column.columns,
            'importance': np.abs(shap_values_for_column).mean(axis=0)
        }).sort_values('importance', ascending=False)

        # Apply multi-criteria feature selection
        selected_important_features = apply_multicriteria_feature_selection(
            shap_importances_df,
            min_absolute_importance=0.1,
            domain_keywords=['Chiral', 'chiral']
        )
        logger.info(f"Selected {len(selected_important_features)} features for '{column_name}' "
                    f"based on multi-criteria approach.")

        # Generate selection summary for reporting
        elbow_idx_report, elbow_threshold_report = determine_elbow_point(
            shap_importances_df['importance'].sort_values(ascending=False).values
        )
        selection_summary = {
            'elbow_index': elbow_idx_report,
            'elbow_threshold': elbow_threshold_report,
            'min_absolute_importance_threshold': 0.1,
            'total_features_evaluated': len(shap_importances_df),
            'selected_features_count': len(selected_important_features)
        }

        # Store results for cross-column comparison
        analysis_results[column_name] = {
            'full_shap_importances': shap_importances_df,
            'selected_features_df': selected_important_features,
            'selection_info': selection_summary,
            'shap_values': shap_values_for_column,
            'X_data': X_column
        }

        # Save detailed feature selection report
        report_path = f"{result_directory}/feature_selection_report_{column_name}.txt"
        with open(report_path, "w") as f:
            f.write(f"Comprehensive Feature Selection Report for Column: '{column_name}'\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total features evaluated: {selection_summary['total_features_evaluated']}\n")
            f.write(f"Number of features selected: {selection_summary['selected_features_count']}\n")
            f.write(f"Elbow point identified at rank: {selection_summary['elbow_index'] + 1} "
                    f"(Importance: {selection_summary['elbow_threshold']:.6f})\n")
            f.write(f"Absolute importance threshold applied: "
                    f"{selection_summary['min_absolute_importance_threshold']:.6f}\n")
            f.write("\nSelected Features (Ranked by SHAP Importance):\n")
            f.write(selected_important_features[['feature', 'importance', 'importance_rank']].to_string(index=False))

        logger.info(f"Feature selection report saved to: {report_path}")

        # Generate visualizations for current column
        plot_importance_curve(
            shap_importances_df['importance'].sort_values(ascending=False).reset_index(drop=True),
            elbow_idx_report, elbow_threshold_report, column_name, result_directory,
            title_fontsize=title_fontsize, label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize, legend_fontsize=legend_fontsize
        )

        plot_shap_summary(
            shap_values_for_column, X_column, selected_important_features, column_name, result_directory,
            title_fontsize=title_fontsize, shap_summary_feature_fontsize=shap_summary_feature_fontsize
        )

        # Save selected features to CSV
        selected_important_features.to_csv(f"{result_directory}/selected_important_features_{column_name}.csv",
                                           index=False)
        logger.info(f"Selected important features list saved for '{column_name}'.")

    # Step 9: Cross-Column Comparison and Combined Visualization
    if analysis_results:
        # Generate cross-column feature comparison
        logger.info("\nInitiating cross-column feature importance comparison...")
        inter_column_comparison_df = pd.DataFrame()

        for col_name, res in analysis_results.items():
            features_for_comparison = res['selected_features_df'].copy()
            features_for_comparison['Column'] = col_name
            inter_column_comparison_df = pd.concat([inter_column_comparison_df, features_for_comparison])

        inter_column_comparison_df.to_csv("Output/SHAP_Results/comparison_of_selected_features.csv", index=False)
        logger.info("Cross-column selected features comparison saved to "
                    "'Output/SHAP_Results/comparison_of_selected_features.csv'.")

        # Prepare data for heatmap visualization
        all_unique_features = sorted(list(inter_column_comparison_df['feature'].unique()))
        heatmap_data_matrix = pd.DataFrame(0.0, index=all_unique_features, columns=unique_columns, dtype=float)

        # Populate heatmap matrix with importance values
        for col_name in unique_columns:
            if col_name not in analysis_results:
                continue

            full_importances_for_column = analysis_results[col_name]['full_shap_importances'].set_index('feature')
            selected_features_for_column = analysis_results[col_name]['selected_features_df']['feature']

            for feature_name in selected_features_for_column:
                if feature_name in full_importances_for_column.index:
                    importance_value = full_importances_for_column.loc[feature_name, 'importance']
                    heatmap_data_matrix.loc[feature_name, col_name] = importance_value

        # Generate cross-column comparison visualizations
        plot_heatmap_of_feature_importances(
            heatmap_data_matrix, unique_columns, all_unique_features,
            title_fontsize=title_fontsize, heatmap_cbar_fontsize=heatmap_cbar_fontsize,
            heatmap_tick_fontsize=heatmap_tick_fontsize
        )

        # Generate combined circular SHAP summary plot
        logger.info("\nGenerating combined circular SHAP summary plot for all columns...")
        column_colors = {
            'Cyclosil_B': '#1f77b4',
            'Cyclodex_B': '#ff7f0e',
            'HP_chiral_20β': '#2ca02c',
            'CP_Cyclodextrin_β_2,3,6_M_19': '#d62728',
            'CP_Chirasil_D_Val': '#9467bd',
            'CP_Chirasil_Dex_CB': '#8c564b',
            'CP_Chirasil_L_Val': '#e377c2'
        }

        plot_multi_column_circular_summary(
            analysis_results,
            column_colors,
            "Output/SHAP_Results"
        )

    else:
        logger.warning("No sufficient data available across columns for comprehensive analysis.")

    logger.info("Overall analysis concluded. All results are stored in the 'SHAP_Results' directory.")


def plot_importance_curve(
        sorted_importances, elbow_idx, elbow_threshold, column_name, result_directory,
        title_fontsize=16, label_fontsize=14, tick_fontsize=12, legend_fontsize=10
):
    """
    Generate and save feature importance curve with elbow point identification.

    This plot visualizes the distribution of feature importances and highlights
    the automatically identified elbow point for feature selection.

    Args:
        sorted_importances (pd.Series): Sorted feature importances (descending order)
        elbow_idx (int): Index of the identified elbow point
        elbow_threshold (float): Importance value at the elbow point
        column_name (str): Name of the chromatographic column being analyzed
        result_directory (str): Directory path for saving the plot
        title_fontsize (int): Font size for plot title
        label_fontsize (int): Font size for axis labels
        tick_fontsize (int): Font size for tick labels
        legend_fontsize (int): Font size for legend
    """
    with plot_context(figsize=(7, 5)) as fig:
        # Plot the importance curve
        plt.plot(sorted_importances, 'b-', linewidth=2, label='SHAP Importance')

        # Highlight the elbow point
        plt.axvline(x=elbow_idx, color='r', linestyle='--',
                    label=f'Elbow Point (Rank: {elbow_idx + 1}, Value: {elbow_threshold:.4f})')
        plt.scatter(elbow_idx, elbow_threshold, c='red', s=100, zorder=5, edgecolors='black')

        # Customize plot appearance
        plt.xlabel("Feature Rank", fontsize=label_fontsize)
        plt.ylabel("Mean Absolute SHAP Importance", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(result_directory, f"feature_importance_elbow_plot_{column_name}.png")
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        logger.info(f"Feature importance curve saved to: {plot_path}")


def plot_shap_summary(
        shap_values, X_test_data, selected_features_df, column_name, result_directory,
        title_fontsize=16, shap_summary_feature_fontsize=10, top_n_features=20
):
    """
    Generate and save SHAP summary plot for selected important features.

    This plot provides a detailed view of how each feature contributes to model
    predictions, showing both the magnitude and direction of SHAP values.

    Args:
        shap_values (np.ndarray): SHAP values for all features and samples
        X_test_data (pd.DataFrame): Feature matrix for the column subset
        selected_features_df (pd.DataFrame): DataFrame of selected important features
        column_name (str): Name of the chromatographic column being analyzed
        result_directory (str): Directory path for saving the plot
        title_fontsize (int): Font size for plot title
        shap_summary_feature_fontsize (int): Font size for feature names in SHAP plot
        top_n_features (int): Maximum number of features to display
    """
    # Limit to the top N features for visualization clarity
    top_n_selected_features_df = selected_features_df.head(top_n_features)
    selected_feature_names = top_n_selected_features_df['feature'].tolist()

    # Ensure feature names exist in the data columns
    valid_feature_names = [f for f in selected_feature_names if f in X_test_data.columns]
    selected_indices = [X_test_data.columns.get_loc(f) for f in valid_feature_names]

    # Filter SHAP values and feature data to include only selected features
    filtered_shap_values = shap_values[:, selected_indices]
    filtered_X_test_data = X_test_data[valid_feature_names]

    with plot_context(figsize=(12, max(8, len(valid_feature_names) * 0.4))) as fig:
        # Generate SHAP summary plot
        shap.summary_plot(filtered_shap_values, filtered_X_test_data,
                          feature_names=valid_feature_names, show=False, plot_size="auto")

        # Customize plot appearance
        ax = plt.gca()
        ax.set_yticklabels(valid_feature_names, fontsize=shap_summary_feature_fontsize)
        ax.set_xlabel("SHAP Value", fontsize=shap_summary_feature_fontsize, fontweight='bold')
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(result_directory,
                                 f"shap_summary_top{len(valid_feature_names)}_features_plot_{column_name}.png")
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        logger.info(f"SHAP summary plot for top {len(valid_feature_names)} features saved to: {plot_path}")


def _simple_beeswarm_jitter(values, n_bins=100):
    """
    Calculate simple jitter for beeswarm plots to spread overlapping points.

    This function distributes points with similar values horizontally to avoid
    overlap in scatter plots, creating a beeswarm-like visualization effect.

    Args:
        values (np.ndarray): Array of values to calculate jitter for
        n_bins (int): Number of bins for value discretization

    Returns:
        np.ndarray: Array of jitter values for horizontal displacement
    """
    n_points = len(values)
    if n_points == 0:
        return np.array([])

    # Handle case where all values are identical
    val_min, val_max = np.min(values), np.max(values)
    if val_min == val_max:
        return np.linspace(-1, 1, n_points)

    # Create bins for value discretization
    bins = np.linspace(val_min, val_max, n_bins)
    binned_indices = np.digitize(values, bins)

    # Group points by bin and calculate jitter within each bin
    bin_counts = {i: [] for i in range(n_bins + 2)}
    for i, bin_idx in enumerate(binned_indices):
        bin_counts[bin_idx].append(i)

    jitter = np.zeros(n_points)
    for _, indices in bin_counts.items():
        count = len(indices)
        if count > 1:
            # Distribute points evenly within the bin range [-1, 1]
            bin_jitter = np.linspace(-1, 1, count)
            for i, point_idx in enumerate(indices):
                jitter[point_idx] = bin_jitter[i]

    return jitter


def plot_multi_column_circular_summary(
        analysis_results, column_colors, result_directory, top_n_features=10,
        clip_percentile=99.0
):
    """
    Generate combined multi-column circular SHAP summary visualization.

    This advanced visualization displays SHAP analysis results for all columns
    in a circular polar plot, combining bar charts (mean importance) with 
    scatter plots (individual SHAP values) in a single comprehensive view.

    Args:
        analysis_results (dict): Dictionary containing SHAP analysis results for each column
        column_colors (dict): Color mapping for each chromatographic column
        result_directory (str): Directory path for saving the plot
        top_n_features (int): Number of top features to display per column
        clip_percentile (float): Percentile for clipping extreme SHAP values
    """
    logger.info(f"Generating combined multi-column circular SHAP summary plot, "
                f"clipping at {clip_percentile}th percentile...")

    column_names = list(analysis_results.keys())
    n_columns = len(column_names)
    if n_columns == 0:
        logger.warning("No analysis results to plot.")
        return

    # Step 1: Figure Setup and Layout Configuration
    with plot_context(figsize=(26, 22), facecolor='white') as fig:
        # Define main axis position, leaving space for legend on the right
        ax = fig.add_axes([0.05, 0.05, 0.8, 0.9], projection='polar')
        ax.set_facecolor('white')

        # Step 2: Layout Parameters and Scaling Configuration
        column_angles = np.linspace(0, 2 * np.pi, n_columns, endpoint=False)
        slice_width = (2 * np.pi / n_columns)

        # Collect all data for consistent scaling across columns
        all_mean_shaps = []
        all_indiv_shaps = []
        for col_name in column_names:
            top_features_df = analysis_results[col_name]['selected_features_df'].head(top_n_features)
            if not top_features_df.empty:
                all_mean_shaps.extend(top_features_df['importance'].values)

            feature_names = top_features_df['feature'].tolist()
            if not feature_names:
                continue
            feature_indices = [analysis_results[col_name]['X_data'].columns.get_loc(f) for f in feature_names]
            shap_subset = analysis_results[col_name]['shap_values'][:, feature_indices]
            if shap_subset.size > 0:
                all_indiv_shaps.extend(shap_subset.flatten())

        if not all_mean_shaps or not all_indiv_shaps:
            logger.warning("Not enough data to generate plot. Aborting.")
            return

        # Redefine scaling parameters (enlarged bar chart area)
        bar_zone_radius = 5  # Increased from 3 to 6 for larger bar charts
        point_zone_radius = 3  # Reduced from 1.5 to 1.2 for smaller scatter area  
        gap = 0.1  # Reduced gap from 0.5 to 0.3

        # Inner circle radius (within bar chart area)
        inner_circle_radius = 0.5

        # Baseline position (where SHAP=0 is located)
        baseline_radius = bar_zone_radius + gap

        # Calculate scaling parameters
        overall_max_mean_shap = np.max(all_mean_shaps)
        clip_val = np.percentile(np.abs(all_indiv_shaps), clip_percentile)

        logger.info(f"Clipping individual SHAP values at: +/- {clip_val:.4f}")

        # Step 3: Draw Inner Circle
        circle_theta = np.linspace(0, 2 * np.pi, 100)
        circle_r = np.full_like(circle_theta, inner_circle_radius)
        ax.plot(circle_theta, circle_r, color='gray', linewidth=2, zorder=15)

        # Step 4: Loop Through Each Column to Draw Slices
        for i, col_name in enumerate(column_names):
            column_angle_start = column_angles[i]

            # Current column data
            data = analysis_results[col_name]
            top_features_df = data['selected_features_df'].head(top_n_features)
            feature_names = top_features_df['feature'].tolist()
            n_features = len(feature_names)
            if n_features == 0:
                continue

            feature_indices = [data['X_data'].columns.get_loc(f) for f in feature_names]
            shap_values_subset = data['shap_values'][:, feature_indices]
            X_data_subset = data['X_data'][feature_names]
            mean_abs_shaps = top_features_df['importance'].values

            # Angular distribution of features within slice
            feature_angles = np.linspace(
                column_angle_start + slice_width * 0.05,
                column_angle_start + slice_width * 0.95,
                n_features, endpoint=True
            )
            feature_bar_width = (slice_width / n_features) * 0.6

            # Draw inner bar chart (mean absolute SHAP values)
            # Bar chart starts from outside of inner circle
            scaled_bars = (mean_abs_shaps / overall_max_mean_shap) * (bar_zone_radius - inner_circle_radius)
            bar_bottom = inner_circle_radius  # Bar chart starting position

            ax.bar(feature_angles, scaled_bars, width=feature_bar_width,
                   bottom=bar_bottom,  # Start from outside inner circle
                   color=column_colors.get(col_name, 'gray'), alpha=0.6,
                   zorder=10)

            # Draw outer beeswarm plot (individual SHAP values)
            cmap = mcolors.LinearSegmentedColormap.from_list("blue_purple_red", ["#0000FF", "#800080", "#FF0000"])
            for j in range(n_features):
                angle = feature_angles[j]
                shap_vals_feature = shap_values_subset[:, j]
                # Clip SHAP values for visualization
                clipped_shap_vals = np.clip(shap_vals_feature, -clip_val, clip_val)

                # Scale points to point zone area
                scaled_points = (clipped_shap_vals / clip_val) * point_zone_radius
                radii = baseline_radius + scaled_points

                feature_vals = X_data_subset.iloc[:, j].values
                scaler = MinMaxScaler()
                feature_colors_normalized = scaler.fit_transform(feature_vals.reshape(-1, 1)).flatten()
                colors = cmap(feature_colors_normalized)

                max_jitter_angle = (feature_bar_width / 2) * 0.8
                jitter = _simple_beeswarm_jitter(clipped_shap_vals) * max_jitter_angle
                jittered_angles = angle + jitter

                ax.scatter(jittered_angles, radii, c=colors, s=15, alpha=0.7, edgecolor='none', zorder=5)

            # Add feature labels
            for j, name in enumerate(feature_names):
                angle = feature_angles[j]
                rotation = np.rad2deg(angle)
                if np.pi / 2 < angle < 3 * np.pi / 2:
                    rotation += 180
                ax.text(angle, baseline_radius + point_zone_radius * 1.15, name,
                        rotation=rotation, ha='center', va='center', fontsize=18, weight='bold',
                        rotation_mode="anchor")

        # Step 5: Aesthetic Settings and Global Labels
        ax.set_rlim(0, baseline_radius + point_zone_radius * 1.1)
        ax.set_rorigin(0)

        # Simplify radial grid, showing only baseline
        ax.set_rticks([baseline_radius])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.grid(color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_color('black')
        ax.spines['polar'].set_linestyle('--')
        ax.spines['polar'].set_linewidth(0.8)

        # Step 6: Color Bar Legend
        cax = fig.add_axes([0.90, 0.22, 0.04, 0.75])
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label('Normalized Feature Value', size=20, weight='bold', labelpad=20)
        cb.ax.tick_params(labelsize=18)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['Low', 'High'], weight='bold')

        # Step 7: Column Legend
        lax = fig.add_axes([0.80, 0.03, 0.20, 0.4])
        lax.axis('off')
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=column_colors.get(name, 'black'))
                          for name in column_names]
        lax.legend(legend_handles, column_names,
                   title="Chromatographic Columns",
                   loc='lower left',
                   prop={'size': 18, 'weight': 'bold'},
                   title_fontproperties={'size': 20, 'weight': 'bold'},
                   frameon=False)

        # Step 8: Save Figure
        plot_path = os.path.join(result_directory, "shap_circular_summary_plot_ALL_COLUMNS.png")
        plt.savefig(plot_path, dpi=600, facecolor='white', bbox_inches='tight')
        logger.info(f"Combined circular SHAP summary plot saved to: {plot_path}")


def plot_heatmap_of_feature_importances(
        heatmap_data, column_names, feature_names,
        title_fontsize=16, heatmap_cbar_fontsize=10, heatmap_tick_fontsize=8
):
    """
    Generate and save heatmap visualizing feature importance across columns.

    This visualization provides a matrix view of how different molecular features
    rank in importance across various chromatographic columns, enabling easy
    identification of universally important features and column-specific patterns.

    Args:
        heatmap_data (pd.DataFrame): Matrix of feature importances (features x columns)
        column_names (list): List of chromatographic column names
        feature_names (list): List of molecular feature names
        title_fontsize (int): Font size for plot title
        heatmap_cbar_fontsize (int): Font size for colorbar labels
        heatmap_tick_fontsize (int): Font size for axis tick labels
    """
    max_features = 20  # Limit visualization to top features for readability
    logger.info(f"Total unique features: {len(feature_names)}")

    if len(feature_names) > max_features:
        logger.warning(f"Too many features ({len(feature_names)}). "
                       f"Limiting heatmap to top {max_features} features.")
        # Select top features by summed importance across all columns
        feature_sums = heatmap_data.sum(axis=1)
        top_features = feature_sums.nlargest(max_features).index
        heatmap_data = heatmap_data.loc[top_features]
        feature_names = top_features.tolist()

    with plot_context(figsize=(12, max(8, len(feature_names) * 0.4))) as fig:
        # Create heatmap using matplotlib
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')

        # Add colorbar with custom formatting
        cbar = plt.colorbar(label='SHAP Importance (Mean Absolute)')
        cbar.ax.tick_params(labelsize=heatmap_cbar_fontsize)

        # Customize axis labels and ticks
        plt.xticks(range(len(column_names)), column_names, rotation=45, ha='right',
                   fontsize=heatmap_tick_fontsize)
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index,
                   fontsize=heatmap_tick_fontsize)
        plt.xlabel("Chromatographic Column", fontsize=14)
        plt.ylabel("Molecular Feature", fontsize=14)
        plt.tight_layout()

        # Save heatmap
        heatmap_path = "Output/SHAP_Results/selected_features_heatmap.png"
        plt.savefig(heatmap_path, dpi=600, bbox_inches='tight')
        logger.info(f"Heatmap of feature importances across columns saved to: {heatmap_path}")


if __name__ == "__main__":
    """
    Main execution block for SHAP analysis pipeline.

    This block ensures required directories exist and initiates the complete
    SHAP analysis workflow with optimized font sizes and visualization parameters.
    """
    # Ensure the base results directory exists
    if not os.path.exists("SHAP_Results"):
        os.makedirs("SHAP_Results")
        logger.info("Created base results directory: 'SHAP_Results'")

    # Verify dataset directory exists
    if not os.path.exists("dataset"):
        logger.error("Dataset directory not found. Please create a 'dataset' folder with 'data.csv'.")
    else:
        # Execute main analysis with optimized visualization parameters
        run_shap_analysis(
            title_fontsize=18,  # Larger titles for better readability
            label_fontsize=16,  # Clear axis labels
            tick_fontsize=14,  # Readable tick labels
            legend_fontsize=14,  # Prominent legends
            shap_summary_feature_fontsize=10,  # Compact feature names in SHAP plots
            heatmap_cbar_fontsize=14,  # Clear colorbar labels
            heatmap_tick_fontsize=10  # Readable heatmap ticks
        )
        logger.info("Analysis script finished execution.")