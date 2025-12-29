"""Molecular Clustering Analysis Pipeline

This script performs molecular similarity analysis and clustering based on chemical fingerprints.
Designed for analyzing chiral molecule retention patterns in chromatography data.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from rdkit import Chem
from rdkit.Chem import AllChem, RemoveStereochemistry
from rdkit.DataStructs import TanimotoSimilarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configuration constants for fingerprint generation and clustering
FINGERPRINT_CONFIG = {
    "radius": 3,  # Morgan fingerprint radius (controls atomic environment size)
    "n_bits": 2048  # Bit vector length (affects fingerprint resolution)
}

CLUSTERING_CONFIG = {
    "max_k": 6,  # Maximum number of clusters to consider
    "silhouette_threshold": 0.5  # Minimum acceptable silhouette score
}


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess molecular data from CSV file.

    Args:
        file_path: Path to input CSV file containing SMILES strings and column info

    Returns:
        Preprocessed DataFrame with canonical SMILES

    Raises:
        AssertionError: If required columns are missing
    """
    data = pd.read_csv(file_path)

    # Validate required columns presence
    required_columns = {"Chiral_molecules_smile", "Column"}
    assert required_columns.issubset(data.columns), \
        f"Missing columns: {required_columns - set(data.columns)}"

    # Generate canonical SMILES without stereochemistry
    data["Canonical_SMILES"] = data["Chiral_molecules_smile"].apply(
        generate_canonical_smiles
    )

    return data.dropna(subset=["Canonical_SMILES"])


def generate_canonical_smiles(smiles: str) -> str:
    """Generate standardized SMILES without stereochemical information.

    Process flow:
    1. Parse SMILES to molecular structure
    2. Remove stereochemical markers
    3. Generate canonical SMILES string

    Args:
        smiles: Input SMILES string with potential chirality

    Returns:
        str: Canonical SMILES without chirality or None for invalid inputs
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        RemoveStereochemistry(mol)  # Discard stereochemical information
        return Chem.MolToSmiles(mol, canonical=True)  # Generate canonical form
    return None


def generate_fingerprints(molecules: List[Chem.Mol]) -> List:
    """Generate Morgan fingerprint bit vectors for molecular structures.

    Fingerprint parameters:
    - Radius 3: Captures molecular features within 3 bonds from each atom
    - 2048 bits: Balance between resolution and computational efficiency

    Args:
        molecules: List of valid RDKit molecule objects

    Returns:
        List: Morgan fingerprints as bit vectors
    """
    return [
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=FINGERPRINT_CONFIG["radius"],
            nBits=FINGERPRINT_CONFIG["n_bits"]
        )
        for mol in molecules
    ]


def compute_similarity_matrix(fingerprints: List) -> np.ndarray:
    """Compute pairwise Tanimoto similarity matrix.

    Tanimoto similarity definition:
    Similarity = (A ∩ B) / (A + B - A ∩ B)
    where A and B are fingerprint bit vectors

    Optimization note: Computes upper triangle only and mirrors values
    for O(n^2/2) complexity instead of O(n^2)

    Args:
        fingerprints: List of molecular fingerprints

    Returns:
        np.ndarray: Symmetric similarity matrix [0-1] values
    """
    n = len(fingerprints)
    matrix = np.zeros((n, n))

    # Upper triangle computation with symmetry
    for i in range(n):
        for j in range(i, n):
            similarity = TanimotoSimilarity(fingerprints[i], fingerprints[j])
            matrix[i, j] = matrix[j, i] = similarity  # Mirror values

    return matrix


def determine_optimal_clusters(feature_matrix: np.ndarray) -> int:
    """Determine optimal cluster count using validation metrics.

    Algorithm steps:
    1. Evaluate silhouette score and Davies-Bouldin index for k=2 to max_k
    2. Select best k from silhouette score if above threshold
    3. Fallback to Davies-Bouldin index if silhouette insufficient

    Args:
        feature_matrix: 2D array of fingerprint features

    Returns:
        int: Optimal cluster count (0 = no clustering)
    """
    if len(feature_matrix) < 2:
        return 0  # Insufficient data points

    silhouette_scores = []
    db_indices = []

    # Evaluate cluster quality metrics for k=2 to max_k
    for k in range(2, CLUSTERING_CONFIG["max_k"] + 1):
        if k >= len(feature_matrix):
            break  # Prevent more clusters than data points

        # Cluster and calculate metrics
        kmeans = KMeans(n_clusters=k, random_state=0).fit(feature_matrix)
        labels = kmeans.labels_

        # Handle edge cases for small clusters
        try:
            silhouette_avg = silhouette_score(feature_matrix, labels)
            db_index = davies_bouldin_score(feature_matrix, labels)
        except ValueError:
            continue  # Skip invalid cluster configurations

        silhouette_scores.append(silhouette_avg)
        db_indices.append(db_index)

    if not silhouette_scores:  # No valid cluster configurations
        return 1  # Default to single cluster

    # Determine best candidates from both metrics
    best_silhouette_k = 2 + np.argmax(silhouette_scores)
    best_db_k = 2 + np.argmin(db_indices)

    # Decision logic based on silhouette threshold
    if silhouette_scores[best_silhouette_k - 2] >= CLUSTERING_CONFIG["silhouette_threshold"]:
        final_k = best_silhouette_k
    else:
        final_k = best_db_k

    return min(final_k, CLUSTERING_CONFIG["max_k"])  # Enforce maximum k limit


def analyze_clusters(cluster_labels: np.ndarray, similarity_matrix: np.ndarray) -> List[Dict]:
    """Analyzes cluster results and computes intra-cluster similarity statistics.

    Args:
        cluster_labels: 1D array of integer cluster assignments (n_samples,)
        similarity_matrix: 2D array of pairwise similarity scores (n_samples, n_samples)

    Returns:
        List of dictionaries containing cluster metadata:
        - cluster_id: Unique identifier for the cluster
        - size: Number of molecules in cluster
        - avg_similarity: Mean pairwise similarity within cluster
        - similarities: Array of all intra-cluster similarity values
    """
    results = []
    # Iterate through unique cluster IDs present in labels
    for cluster_id in np.unique(cluster_labels):
        # Get indices of molecules belonging to current cluster
        indices = np.where(cluster_labels == cluster_id)[0]

        # Skip clusters with no members (safety check)
        if len(indices) < 1:
            continue

        # Extract similarity submatrix for current cluster using numpy indexing
        sub_matrix = similarity_matrix[np.ix_(indices, indices)]
        # Extract upper triangle values (excluding diagonal) for pairwise similarities
        triu_values = sub_matrix[np.triu_indices_from(sub_matrix, k=1)]

        results.append({
            "cluster_id": cluster_id,
            "size": len(indices),
            # Handle empty clusters with conditional expression
            "avg_similarity": triu_values.mean() if triu_values.size > 0 else 0.0,
            "similarities": triu_values  # Raw similarity values for visualization
        })
    return results


def save_similarity_matrix(matrix: np.ndarray, category: str):
    """Persists similarity matrix to CSV file for future analysis.

    Args:
        matrix: 2D numpy array of similarity scores
        category: Identifier string for file naming (e.g., column name)
    """
    # Create output directory if not exists (idempotent operation)
    Path("similarity_matrices").mkdir(exist_ok=True)

    # Save as CSV without row/column indices
    pd.DataFrame(matrix).to_csv(
        f"similarity_matrices/{category}_similarity.csv",
        index=False
    )


def visualize_similarity_matrix(matrix: np.ndarray, category: str):
    """Generates publication-quality heatmap visualization of similarity matrix.

    Visualization features:
    - Upper triangle masking to avoid duplicate information
    - Viridis color map for colorblind-friendly perception
    - High-resolution 600dpi output suitable for print
    - Limits axis ticks to 5 values for clarity.

    Args:
        matrix: 2D similarity matrix to visualize
        category: Identifier for plot title and filename
    """
    # Configure global plot aesthetics
    plt.rcParams.update({
        'font.size': 18,  # Base font size
        'axes.titlesize': 18,  # Title size
        'axes.labelsize': 16,  # Axis label size
        'xtick.labelsize': 14,  # X-axis tick labels
        'ytick.labelsize': 14 # Y-axis tick labels
    })

    plt.figure(figsize=(4, 4))  # Square aspect ratio
    ax = sns.heatmap(
        matrix,
        cmap='viridis',  # Perceptually uniform colormap
        square=True,  # Force square cells
        cbar_kws={'label': 'Tanimoto Similarity', 'shrink': 0.7},  # Colorbar label
        annot=False,  # Disable cell annotations for clarity
        mask=np.triu(np.ones_like(matrix, dtype=bool)))  # Mask upper triangle

    # --- 最终解决方案 ---
    # 1. 使用 MaxNLocator 确定刻度的位置
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    # 2. 手动获取刻度位置并强制设置为标签
    # 这是最直接的方法，可以覆盖任何抑制标签的内部设置。
    x_ticks = ax.get_xticks()
    ax.set_xticklabels([int(tick) for tick in x_ticks])

    y_ticks = ax.get_yticks()
    ax.set_yticklabels([int(tick) for tick in y_ticks])
    # --- 解决方案结束 ---


    # Axis labels with padding
    ax.set_xlabel("Molecular Index", labelpad=2)
    ax.set_ylabel("Molecular Index", labelpad=2)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save high-resolution PNG with transparency
    plt.savefig(
        f"similarity_matrices/{category}_heatmap.png",
        transparent=True,  # For figure overlays
        bbox_inches='tight',  # Remove whitespace borders
        dpi=600  # Print-quality resolution
    )
    plt.close()  # Prevent memory leaks in batch processing

def visualize_clusters(cluster_data: List[Dict], category: str):
    """Generates comparative boxplots of intra-cluster similarity distributions.

    Updates:
    - Dynamically adjusts figure width based on cluster count
    - Ensures consistent physical dimensions for content area
    - Fixed box width (0.6) and spacing (0.4) for visual consistency

    Args:
        cluster_data: Analyzed cluster results from analyze_clusters()
        category: Identifier for plot title and filename
    """
    # Configure plot aesthetics
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    # Transform nested cluster data to flat DataFrame
    box_data = []
    for cluster in cluster_data:
        box_data.extend([
            {"Cluster": f"Cluster {cluster['cluster_id']}", "Similarity": sim}
            for sim in cluster["similarities"]
        ])

    if not box_data:
        return  # Skip empty data

    df = pd.DataFrame(box_data)
    n_clusters = len(cluster_data)

    # Calculate dynamic width: 0.8 inches per cluster + 2-inch base
    fig_width = max(8.0, 2.0 + n_clusters * 0.8)  # Minimum 6 inches
    plt.figure(figsize=(fig_width, 6))  # Fixed height

    # Create boxplot with fixed parameters
    ax = sns.boxplot(
        x="Cluster",
        y="Similarity",
        data=df,
        hue="Cluster",
        palette="tab10",
        width=0.6,  # Fixed box width (data units)
        dodge=False,  # Prevent automatic spacing
        linewidth=0.8,
        legend=False
    )

    # Adjust spacing between boxes
    for i, box in enumerate(ax.artists):
        box.set_x(box.get_x() + 0.2 * i)  # Manual spacing adjustment

    # Plot decorations
    ax.set_xlabel('')
    plt.ylabel("Tanimoto Similarity", labelpad=10)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save with consistent content dimensions
    plt.savefig(
        f"cluster_boxplots/{category}_boxplot.png",
        bbox_inches="tight",
        dpi=600,
        transparent=True
    )
    plt.close()


def process_column_data(column_group: pd.DataFrame, column_name: str) -> Tuple[float, List[Dict]]:
    """Processes chromatography column data through full analysis pipeline.

    Workflow stages:
    1. Data deduplication
    2. Molecular validation
    3. Fingerprint generation
    4. Similarity analysis
    5. Clustering optimization
    6. Results storage

    Args:
        column_group: DataFrame of molecules for specific chromatography column
        column_name: Identifier for current column

    Returns:
        Tuple containing:
        - Average similarity score (float)
        - Cluster analysis results (List[Dict] or None)
    """
    # Stage 1: Data deduplication
    unique_group = column_group.drop_duplicates(subset=["Canonical_SMILES"], keep="first")
    if len(unique_group) < 2:
        print(f"Skipped {column_name}: insufficient unique molecules")
        return None, None

    # Stage 2: Molecular validation
    valid_molecules = [mol for mol in (
        Chem.MolFromSmiles(s) for s in unique_group["Chiral_molecules_smile"]
    ) if mol is not None]

    if len(valid_molecules) < 2:
        print(f"Skipped {column_name}: insufficient valid molecules")
        return None, None

    # Stage 3: Fingerprint generation
    fingerprints = generate_fingerprints(valid_molecules)

    # Stage 4: Similarity analysis
    similarity_matrix = compute_similarity_matrix(fingerprints)
    save_similarity_matrix(similarity_matrix, column_name)
    visualize_similarity_matrix(similarity_matrix, column_name)

    # Calculate global average similarity (upper triangle)
    avg_similarity = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)].mean()

    # Stage 5: Clustering analysis
    feature_matrix = np.array([list(fp) for fp in fingerprints])
    optimal_k = determine_optimal_clusters(feature_matrix)

    if optimal_k < 1:  # No meaningful clustering
        return avg_similarity, None

    # Perform K-means clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(feature_matrix)
    cluster_results = analyze_clusters(kmeans.labels_, similarity_matrix)

    return avg_similarity, cluster_results


def main():
    # Directory initialization with exist_ok for idempotency
    Path("similarity_matrices").mkdir(exist_ok=True)
    Path("cluster_boxplots").mkdir(exist_ok=True)

    # Data loading with preprocessing
    data = load_and_preprocess_data("../data.csv")

    # Result containers
    similarity_results = {}  # Column -> average similarity
    clustering_results = {}  # Column -> cluster metadata

    # Process each chromatography column group
    for column_name, column_group in data.groupby("Column"):
        print(f"\nProcessing {column_name}")

        # Core processing pipeline
        avg_sim, clusters = process_column_data(column_group, column_name)
        if avg_sim is None:
            continue  # Skip invalid columns

        # Store and visualize results
        similarity_results[column_name] = avg_sim
        clustering_results[column_name] = clusters
        visualize_clusters(clusters, column_name)

    # Save aggregated results
    pd.DataFrame.from_dict(similarity_results, orient="index",
                           columns=["AverageSimilarity"]).to_csv("average_similarities.csv")

    # Transform clustering results to tabular format
    clustering_records = [
        {
            "Column": column,
            "ClusterID": cluster["cluster_id"],
            "ClusterSize": cluster["size"],
            "AvgSimilarity": cluster["avg_similarity"]
        }
        for column, clusters in clustering_results.items()
        for cluster in clusters
    ]
    pd.DataFrame(clustering_records).to_csv("clustering_results.csv", index=False)


if __name__ == "__main__":
    main()
