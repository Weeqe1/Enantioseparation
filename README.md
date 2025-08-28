# Chromatographic Retention Time Prediction using Graph Attention Networks (GAT)

This project predicts chromatographic retention times (RT) using molecular graphs and various machine learning models, including Graph Attention Networks (GAT). The approach integrates molecular descriptor calculations, SMILES string processing, and feature engineering, followed by model training, evaluation, and external validation.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Dependencies](#dependencies)
* [Setup Instructions](#setup-instructions)
* [Usage](#usage)
* [SHAP Analysis](#shap-analysis)
* [Bayesian Hyperparameter Optimization](#bayesian-hyperparameter-optimization)
* [Training](#training)
* [License](#license)

## Overview

This project builds machine learning models to predict the retention time (RT) of molecules in chromatographic separations. The primary focus is on the application of **Graph Attention Networks (GAT)** using molecular graph representations.

The project is structured as follows:

1. **Dataset Construction**: Construct the dataset by processing molecular graphs and descriptors using `setup_base.py` and `setup_GAT.py`.
2. **Model Training**: Train models using the preprocessed data and evaluate their performance.
3. **External Dataset Validation**: Validate trained models using external datasets.
4. **SHAP Analysis**: Perform SHAP analysis to identify important features for model predictions.
5. **Hyperparameter Tuning**: Use Bayesian optimization for hyperparameter tuning.

## Features

* **Molecular Feature Calculation**: Calculates Mordred molecular descriptors, extracts graph-based features using RDKit, and processes chiral and column molecules for chromatographic analysis.
* **Modeling**: Implements machine learning algorithms including **GAT**, **Linear Regression**, **Support Vector Regression**, **Gradient Boosting**, and **Random Forest**, etc.
* **Hyperparameter Tuning**: Includes **Bayesian optimization** for hyperparameter tuning.
* **SHAP Analysis**: Performs **SHAP** (SHapley Additive exPlanations) analysis for feature importance, with multi-criteria feature selection.
* **External Validation**: Validates models on external datasets.

## Dependencies

Before running the project, you can install the necessary libraries using the following command:

```bash
pip install -r requirements.txt
```

## Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/Weeqe1/Enantioseparation.git
cd Enantioseparation
```

2. **Install dependencies:**

Ensure that all dependencies are installed as mentioned in the Dependencies section.

3. **Dataset**:

   * Prepare your dataset in a `.csv` format with relevant columns for SMILES strings, retention times (RT), and column types.
   * Place the dataset in the `dataset/` directory.

4. **Configuration**:

   * The execution modes are controlled by the `config.py` file. You can select between **Dataset Construction**, **Model Training**, and **External Validation** by setting the `DESK` variable in `config.py` to:

     * `'Dataset_construct'`
     * `'Train'`
     * `'Validate_External'`
   * Each mode uses specific scripts for execution, which are detailed below.

## Usage

1. **Configure the target column**:

   * In `config.py`, set the `TRANSFER_TARGET` variable to the desired chromatographic column. Available options include: `'Cyclosil_B'`, `'Cyclodex_B'`, `'HP_chiral_20β'`, etc.

2. **Run the setup scripts**:

   * Based on the value of `DESK` in `config.py`, execute the corresponding script:

     * **Dataset Construction**: Set `DESK = 'Dataset_construct'` and run:

       ```bash
       python setup_base.py
       python setup_GAT.py
       ```
     * **Model Training**: Set `DESK = 'Train'` and run:

       ```bash
       python setup_base.py
       python setup_GAT.py
       ```
     * **External Dataset Validation**: Set `DESK = 'Validate_External'` and run:

       ```bash
       python setup_base.py
       python setup_GAT.py
       ```

## SHAP Analysis

To perform SHAP analysis and feature selection, run the corresponding Python script:

```bash
python run_shap_analysis.py
```

This script will calculate SHAP values and visualize the importance of various features based on the trained model.

## Bayesian Hyperparameter Optimization

To perform Bayesian optimization for hyperparameter tuning, run:

```bash
python paremeter_tuning_bayes.py
```

This will optimize hyperparameters like learning rate, batch size, and other model-specific parameters to improve performance.

## Training

The model training process is controlled by the configuration file (`config.py`).

The main steps involved are:

1. **Dataset Construction**: Prepares the molecular dataset by generating 3D structures and calculating descriptors.
2. **Model Training**: Trains the selected model (e.g., GAT or other baseline models) using the preprocessed dataset.
3. **Evaluation**: Models are evaluated using metrics like RMSE, MAE, and R².
4. **Results Visualization**: Use matplotlib to visualize model performance. This generates a Hexbin scatter plot (showing predicted versus true retention times, annotated with performance metrics) and an error histogram (showing the distribution of prediction errors).
### Model Architecture

The primary model used in this project is the **Graph Attention Network (GAT)**, which leverages graph-based features to model the relationships between atoms and bonds in molecules. The architecture includes:

* **GATv2Conv layers**: These layers apply attention mechanisms to prioritize important atoms/bonds in the graph.
* **Global Mean Pooling**: Aggregates node-level features to form graph-level representations.
* **MLP Head**: A multi-layer perceptron for final regression predictions.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.




