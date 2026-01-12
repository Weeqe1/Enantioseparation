# GATChiral: Graph Attention Networks for Chromatographic Retention Time Prediction of Chiral Molecules

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.6+-green.svg)](https://pytorch-geometric.readthedocs.io/)

---

## Abstract

Accurate prediction of chromatographic retention time (RT) for chiral molecules remains a fundamental challenge in analytical chemistry and pharmaceutical development. Traditional Quantitative Structure-Retention Relationship (QSRR) models rely on handcrafted molecular descriptors that inadequately capture the three-dimensional stereochemical interactions governing enantioseparation. We present **GATChiral**, a novel deep learning framework that leverages Graph Attention Networks with dynamic attention mechanisms (GATv2) to model molecular structures as attributed graphs. Our approach integrates: (1) **3D geometric molecular representations** with MMFF-optimized conformations, (2) **dual-molecule feature fusion** capturing both analyte and stationary phase characteristics, and (3) **SHAP-guided feature selection** for interpretable predictions. Experimental results across seven distinct chiral chromatographic columns demonstrate superior predictive performance compared to conventional machine learning baselines, achieving state-of-the-art accuracy in retention time prediction.

---

## Table of Contents

- [1. Introduction and Scientific Background](#1-introduction-and-scientific-background)
- [2. Key Innovations](#2-key-innovations)
- [3. Methodology and Algorithmic Framework](#3-methodology-and-algorithmic-framework)
  - [3.1 Problem Formulation](#31-problem-formulation)
  - [3.2 Molecular Graph Representation](#32-molecular-graph-representation)
  - [3.3 3D Geometric Feature Extraction](#33-3d-geometric-feature-extraction)
  - [3.4 Graph Attention Network Architecture](#34-graph-attention-network-architecture)
  - [3.5 Why GATv2? Theoretical Justification](#35-why-gatv2-theoretical-justification)
  - [3.6 Dual-Molecule Feature Fusion Strategy](#36-dual-molecule-feature-fusion-strategy)
  - [3.7 SHAP-Based Feature Selection](#37-shap-based-feature-selection)
- [4. System Architecture](#4-system-architecture)
- [5. Installation](#5-installation)
- [6. Usage](#6-usage)
- [7. Experimental Results](#7-experimental-results)
- [8. Project Structure](#8-project-structure)
- [9. References and Citation](#9-references-and-citation)

---

## 1. Introduction and Scientific Background

### 1.1 The Enantioseparation Challenge

Chirality is a fundamental molecular property with profound implications in pharmaceutical sciences. Enantiomers—non-superimposable mirror images of chiral molecules—often exhibit dramatically different biological activities, pharmacokinetics, and toxicological profiles. The thalidomide tragedy of the 1960s serves as a stark reminder: while (*R*)-thalidomide provides therapeutic sedation, (*S*)-thalidomide causes severe teratogenic effects.

Gas chromatography (GC) with chiral stationary phases (CSPs) remains the gold standard for enantioseparation. However, method development is traditionally empirical, requiring extensive trial-and-error experimentation across column types, temperature programs, and carrier gas conditions. **Accurate *a priori* prediction of retention times would revolutionize analytical method development**, enabling rational column selection and optimized separation conditions.

### 1.2 Limitations of Traditional QSRR Approaches

Classical Quantitative Structure-Retention Relationship (QSRR) models suffer from several fundamental limitations:

| Limitation | Description |
|------------|-------------|
| **Descriptor Inadequacy** | 2D molecular descriptors fail to capture the 3D stereochemical interactions essential for chiral recognition |
| **Feature Engineering Burden** | Manual selection of relevant descriptors requires domain expertise and is inherently biased |
| **Linear Assumptions** | Traditional models (MLR, PLS) assume linear relationships, poorly approximating complex molecular interactions |
| **Single-Molecule Focus** | Ignoring stationary phase molecular characteristics limits predictive transferability |

### 1.3 Our Solution: Graph Neural Networks for Molecular Modeling

Graph Neural Networks (GNNs) have emerged as the paradigm of choice for molecular property prediction, naturally encoding the topology of molecular structures. Atoms are represented as nodes, chemical bonds as edges, and message-passing mechanisms enable the learning of hierarchical structural representations.

We extend this paradigm with:
- **GATv2 dynamic attention** for learning task-specific atomic importance
- **3D geometric features** encoding stereochemical information
- **Dual-molecule modeling** capturing analyte-CSP interactions

---

## 2. Key Innovations

This work introduces several methodological advances:

> **Innovation 1: GATv2-Based Molecular Encoding**  
> We employ GATv2Conv layers with dynamic attention computation, enabling the model to learn context-dependent atomic importance weights that adapt to the specific prediction task.

> **Innovation 2: 3D Geometric Molecular Graphs**  
> Beyond topological connectivity, we integrate MMFF-optimized 3D coordinates to compute bond lengths and bond angles, providing critical stereochemical information.

> **Innovation 3: Dual-Molecule Feature Fusion**  
> We simultaneously encode both the chiral analyte molecule AND the stationary phase selector molecule, capturing the molecular recognition events underlying enantioseparation.

> **Innovation 4: Residual-Enhanced Deep Architecture**  
> Skip connections with learned linear projections enable training of deeper networks while mitigating gradient degradation.

> **Innovation 5: SHAP-Guided Interpretable Predictions**  
> Multi-criteria feature selection based on SHapley Additive exPlanations provides model interpretability and dimensionality reduction.

> **Innovation 6: MLP Predictor Head**  
> A three-layer multilayer perceptron replaces naive linear readout, providing enhanced non-linear transformation capacity for final regression.

---

## 3. Methodology and Algorithmic Framework

### 3.1 Problem Formulation

Let $\mathcal{M} = \{m_1, m_2, \ldots, m_N\}$ denote a set of chiral molecules, each represented by its SMILES string. For a given chromatographic column $C$ with stationary phase molecule $s_C$, and temperature program $\tau$, our objective is to learn a mapping:

$$f: (\mathcal{G}_m, \mathcal{G}_s, \tau) \rightarrow \hat{y} \in \mathbb{R}^+$$

where $\mathcal{G}_m$ and $\mathcal{G}_s$ are graph representations of the analyte and stationary phase molecules respectively, and $\hat{y}$ is the predicted retention time in minutes.

### 3.2 Molecular Graph Representation

Each molecule $m$ is represented as an attributed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X}, \mathbf{E})$ where:

- $\mathcal{V} = \{v_1, \ldots, v_n\}$: Set of atoms (nodes)
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$: Set of chemical bonds (edges)
- $\mathbf{X} \in \mathbb{R}^{n \times d_v}$: Node feature matrix
- $\mathbf{E} \in \mathbb{R}^{|\mathcal{E}| \times d_e}$: Edge feature matrix

#### Node Features ($d_v = 14$)

| Feature | Description | Encoding |
|---------|-------------|----------|
| `atomic_num` | Atomic number (1-118) | Categorical |
| `chiral_tag` | CIP stereochemistry (R/S/None) | Categorical |
| `degree` | Number of bonded neighbors | Categorical |
| `hybridization` | Orbital hybridization (sp/sp²/sp³) | Categorical |
| `formal_charge` | Formal charge (-5 to +10) | Categorical |
| `is_aromatic` | Aromaticity flag | Binary |
| `num_Hs` | Total hydrogen count | Categorical |
| `is_in_ring` | Ring membership | Binary |
| `valence_out_shell` | Outer shell electrons | Categorical |
| `num_radical_e` | Radical electrons | Categorical |
| `explicit_valence` | Explicit valence | Categorical |
| `implicit_valence` | Implicit valence | Categorical |
| `mass` | Atomic mass | Continuous |
| `vdw_radius` | Van der Waals radius | Continuous |

#### Edge Features ($d_e = 6 + d_{desc} + d_{\tau}$)

| Feature | Description | Encoding |
|---------|-------------|----------|
| `bond_type` | Single/Double/Triple/Aromatic | Categorical |
| `bond_stereo` | E/Z stereochemistry | Categorical |
| `is_conjugated` | Conjugation status | Binary |
| `is_in_ring` | Ring membership | Binary |
| `bond_dir` | Bond direction | Categorical |
| `bond_length` | 3D Euclidean distance | Continuous |
| `descriptors` | Mordred molecular descriptors | Continuous |
| `temperature_program` | GC temperature parameters | Continuous |

### 3.3 3D Geometric Feature Extraction

Stereochemistry is inherently three-dimensional. We generate optimized 3D conformations using the Merck Molecular Force Field (MMFF94):

```
Algorithm: MMFF 3D Conformation Generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: SMILES string s
Output: Optimized 3D coordinates {(x_i, y_i, z_i)}

1. mol ← RDKit.MolFromSmiles(s)
2. mol_H ← AddHydrogens(mol)
3. conformers ← EmbedMultipleConfs(mol_H, numConfs=10)
4. energies ← MMFFOptimizeMoleculeConfs(mol_H)
5. best_idx ← argmin(energies)
6. coords ← GetConformerPositions(best_idx)
7. return RemoveHydrogens(mol), coords
```

From the optimized geometry, we extract:

**Bond Lengths:**
$$d_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|_2 = \sqrt{(x_i-x_j)^2 + (y_i-y_j)^2 + (z_i-z_j)^2}$$

**Bond Angles:**
For three atoms $i$-$j$-$k$ forming an angle at central atom $j$:
$$\theta_{ijk} = \arccos\left(\frac{\mathbf{v}_{ji} \cdot \mathbf{v}_{jk}}{\|\mathbf{v}_{ji}\| \|\mathbf{v}_{jk}\|}\right)$$

where $\mathbf{v}_{ji} = \mathbf{r}_i - \mathbf{r}_j$ and $\mathbf{v}_{jk} = \mathbf{r}_k - \mathbf{r}_j$.

### 3.4 Graph Attention Network Architecture

Our model architecture consists of stacked GATv2 convolutional layers with residual connections, followed by global pooling and an MLP predictor head.

#### 3.4.1 GATv2 Convolution Layer

The Graph Attention mechanism computes attention-weighted message passing. For node $i$, the updated representation $\mathbf{h}'_i$ is:

$$\mathbf{h}'_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j$$

where $\alpha_{ij}$ are learned attention coefficients and $\mathbf{W}$ is a learnable weight matrix.

**GATv2 Dynamic Attention:**

Unlike the original GAT which computes static attention:
$$\alpha_{ij}^{\text{GAT}} = \text{softmax}_j\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)\right)$$

GATv2 employs *dynamic* attention:
$$\alpha_{ij}^{\text{GATv2}} = \text{softmax}_j\left(\mathbf{a}^\top \text{LeakyReLU}\left(\mathbf{W}[\mathbf{h}_i \| \mathbf{h}_j]\right)\right)$$

This subtle reordering moves the non-linearity *inside* the attention mechanism, enabling the network to compute a strictly more expressive class of attention functions.

#### 3.4.2 Multi-Head Attention

We employ $K$ attention heads, concatenating their outputs:
$$\mathbf{h}'_i = \Big\|_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j$$

For the final layer, we average instead of concatenating:
$$\mathbf{h}'_i = \frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j$$

#### 3.4.3 Edge-Conditioned Attention

We extend attention computation to incorporate edge features $\mathbf{e}_{ij}$:
$$\alpha_{ij} = \text{softmax}_j\left(\mathbf{a}^\top \text{LeakyReLU}\left(\mathbf{W}_n[\mathbf{h}_i \| \mathbf{h}_j] + \mathbf{W}_e \mathbf{e}_{ij}\right)\right)$$

This allows the model to modulate attention based on bond properties (type, length, stereochemistry).

#### 3.4.4 Residual Connections and Normalization

For layer $\ell$, we apply:
$$\mathbf{H}^{(\ell+1)} = \text{Dropout}\left(\text{ELU}\left(\text{BatchNorm}\left(\text{GATv2}^{(\ell)}(\mathbf{H}^{(\ell)})\right)\right)\right) + \mathbf{W}_{\text{res}}^{(\ell)} \mathbf{H}^{(\ell)}$$

The residual projection $\mathbf{W}_{\text{res}}^{(\ell)}$ handles dimension mismatches between layers.

#### 3.4.5 Global Graph Pooling

We obtain a graph-level representation via global mean pooling:
$$\mathbf{h}_{\mathcal{G}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathbf{h}_i^{(L)}$$

where $\mathbf{h}_i^{(L)}$ is the final-layer node embedding.

#### 3.4.6 MLP Predictor Head

The graph embedding is processed through a three-layer MLP:
$$\hat{y} = \mathbf{W}_3 \cdot \text{ReLU}\left(\text{Dropout}\left(\mathbf{W}_2 \cdot \text{ReLU}\left(\text{Dropout}\left(\mathbf{W}_1 \cdot \mathbf{h}_{\mathcal{G}}\right)\right)\right)\right)$$

with progressive dimensionality reduction: $d_{\text{out}} \rightarrow d_{\text{hidden}} \rightarrow d_{\text{hidden}}/2 \rightarrow 1$.

### 3.5 Why GATv2? Theoretical Justification

The selection of GATv2 over alternative GNN architectures is grounded in several theoretical and empirical considerations:

#### 3.5.1 Expressivity Analysis

**Theorem (Brody et al., 2022):** *Standard GAT computes a limited form of "static" attention that cannot rank node importance differently for each query node. GATv2 computes "dynamic" attention that is strictly more expressive.*

For enantioseparation prediction, this is critical: the importance of a chiral center depends on its electronic and steric context. A static attention function cannot capture that the same @@ stereocenter may have different importance in different molecular environments.

#### 3.5.2 Edge Feature Integration

Unlike vanilla GCN or GraphSAGE, GAT architectures naturally extend to edge-conditioned attention. This is essential for our application because:
- **Bond type** affects electron distribution and molecular flexibility
- **Bond length** directly encodes 3D geometry
- **Stereochemistry markers** (E/Z) provide configurational information

#### 3.5.3 Comparison with Alternatives

| Architecture | Edge Features | Attention | 3D Geometry | Suitability |
|--------------|---------------|-----------|-------------|-------------|
| GCN | ✗ | ✗ | ✗ | Low |
| GraphSAGE | ✗ | ✗ | ✗ | Low |
| GAT | ✓ | Static | ✗ | Medium |
| **GATv2** | **✓** | **Dynamic** | **✓** | **High** |
| SchNet | ✓ | ✗ | ✓ | Medium |
| DimeNet | ✓ | ✗ | ✓ | High (but complex) |

GATv2 provides the optimal balance of expressivity, efficiency, and interpretability for our application.

### 3.6 Dual-Molecule Feature Fusion Strategy

A critical insight of our approach is that enantioseparation is fundamentally a *molecular recognition* problem involving two molecules: the chiral analyte and the stationary phase selector.

We compute Mordred descriptors for both molecules:
- $\mathbf{d}_{\text{chiral}} \in \mathbb{R}^{p}$: Analyte descriptors
- $\mathbf{d}_{\text{column}} \in \mathbb{R}^{q}$: Stationary phase descriptors

These are concatenated and broadcast to all edges:
$$\mathbf{e}'_{ij} = [\mathbf{e}_{ij} \| \mathbf{d}_{\text{chiral}} \| \mathbf{d}_{\text{column}} \| \boldsymbol{\tau}]$$

where $\boldsymbol{\tau}$ encodes the GC temperature program.

This fusion strategy enables the model to learn how specific analyte-CSP pairs interact, rather than treating the column as a categorical variable.

### 3.7 SHAP-Based Feature Selection

To ensure interpretability and reduce overfitting, we employ SHapley Additive exPlanations (SHAP) for feature selection:

**Multi-Criteria Selection Algorithm:**
1. Train surrogate XGBoost model on full feature set
2. Compute SHAP values for each feature
3. Rank features by mean absolute SHAP value
4. Apply elbow method to identify importance threshold
5. Filter features below minimum importance threshold
6. Retain features with domain-relevant keywords

This produces column-specific feature subsets, enabling interpretable predictions and computational efficiency.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GATChiral System Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   SMILES     │    │  Temperature │    │   Column     │                   │
│  │   Input      │    │   Program    │    │   Selection  │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         ▼                   │                   ▼                            │
│  ┌──────────────┐           │           ┌──────────────┐                    │
│  │ 3D Conformer │           │           │  Stationary  │                    │
│  │  Generation  │           │           │ Phase SMILES │                    │
│  │   (MMFF94)   │           │           │              │                    │
│  └──────┬───────┘           │           └──────┬───────┘                    │
│         │                   │                   │                            │
│         ▼                   │                   ▼                            │
│  ┌──────────────┐           │           ┌──────────────┐                    │
│  │    Graph     │           │           │   Mordred    │                    │
│  │ Construction │           │           │ Descriptors  │                    │
│  └──────┬───────┘           │           └──────┬───────┘                    │
│         │                   │                   │                            │
│         └───────────────────┼───────────────────┘                            │
│                             │                                                │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    Feature Fusion Layer                       │           │
│  │  [Node Features | Edge Features | Descriptors | Temp Program] │           │
│  └──────────────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    GATv2 Encoder (L layers)                   │           │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │           │
│  │  │ GATv2   │──▶│BatchNorm│──▶│  ELU    │──▶│ Dropout │──┐    │           │
│  │  │ Conv    │   │         │   │         │   │         │  │    │           │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘  │    │           │
│  │       ▲                                                  │    │           │
│  │       └──────────── Residual Connection ─────────────────┘    │           │
│  └──────────────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    Global Mean Pooling                        │           │
│  └──────────────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                      MLP Predictor Head                       │           │
│  │           Linear → ReLU → Dropout → Linear → ReLU → Linear   │           │
│  └──────────────────────────────┬───────────────────────────────┘           │
│                                 │                                            │
│                                 ▼                                            │
│                    ┌────────────────────────┐                               │
│                    │  Predicted RT (minutes) │                               │
│                    └────────────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Installation

### 5.1 Prerequisites

- Python ≥ 3.8
- CUDA ≥ 11.8 (for GPU acceleration)
- 16GB RAM minimum

### 5.2 Environment Setup

```bash
# Clone repository
git clone https://github.com/Weeqe1/Enantioseparation.git
cd Enantioseparation

# Create conda environment
conda create -n gatchiral python=3.10
conda activate gatchiral

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (CUDA 12.6)
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
```

### 5.3 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.6.0 | Deep learning framework |
| `torch-geometric` | ≥2.6.1 | Graph neural networks |
| `rdkit` | ≥2024.9.5 | Molecular processing |
| `mordred` | ≥1.2.0 | Descriptor calculation |
| `shap` | ≥0.48.0 | Feature importance |
| `scikit-learn` | ≥1.6.1 | ML utilities |
| `bayesian-optimization` | ≥2.0.3 | Hyperparameter tuning |

---

## 6. Usage

### 6.1 Configuration

Edit `config.py` to specify execution mode and target column:

```python
DESK = 'Train'              # 'Dataset_construct' | 'Train' | 'Validate_External'
TRANSFER_TARGET = 'Cyclosil_B'  # Target chromatographic column
RANDOM_SEED = 42
```

### 6.2 Supported Chromatographic Columns

| Column ID | Stationary Phase Type | Selector Molecule |
|-----------|----------------------|-------------------|
| `Cyclosil_B` | Cyclodextrin derivative | Permethylated β-CD |
| `Cyclodex_B` | β-Cyclodextrin | Native β-CD |
| `HP_chiral_20β` | Hydroxypropyl-β-CD | HP-β-CD |
| `CP_Chirasil_D_Val` | Amino acid derivative | D-Valine diamide |
| `CP_Chirasil_L_Val` | Amino acid derivative | L-Valine diamide |
| `CP_Chirasil_Dex_CB` | Cyclodextrin composite | Chirasil-Dex |
| `CP_Cyclodextrin_β_2,3,6_M_19` | Methylated β-CD | 2,3,6-tri-O-methyl-β-CD |

### 6.3 Workflow Execution

```bash
# Step 1: Dataset Construction
python setup_GAT.py  # DESK = 'Dataset_construct'

# Step 2: Model Training (5-fold cross-validation)
python setup_GAT.py  # DESK = 'Train'

# Step 3: SHAP Feature Analysis
python run_shap_analysis.py

# Step 4: Hyperparameter Optimization (optional)
python paremeter_tuning_bayes.py

# Step 5: External Validation
python setup_GAT.py  # DESK = 'Validate_External'
```

---

## 7. Experimental Results

### 7.1 Evaluation Metrics

We evaluate model performance using:

- **Mean Absolute Error (MAE):** $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
- **Coefficient of Determination (R²):** $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

### 7.2 Baseline Comparisons

The framework includes comprehensive baseline models:

| Model | Type | Description |
|-------|------|-------------|
| Linear Regression | Linear | Standard OLS regression |
| SVR (RBF) | Kernel | Support vector regression with RBF kernel |
| Gradient Boosting | Ensemble | Gradient boosted decision trees |
| Random Forest | Ensemble | Bootstrap aggregated trees |
| MLP Regressor | Neural | Feedforward neural network |
| Transformer | Attention | Self-attention based regressor |
| **GATv2 (Ours)** | **GNN** | **Graph attention with 3D features** |

### 7.3 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Hidden dimension | 64 |
| Number of layers | 3-4 |
| Attention heads | 4 |
| Dropout rate | 0.2 |
| Learning rate | 1e-3 → 1e-5 |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |
| Loss function | SmoothL1Loss (β=0.5) |
| Cross-validation | 5-fold stratified |
| Early stopping | Patience=100, δ=0.001 |

---

## 8. Project Structure

```
Enantioseparation/
├── GAT_model/
│   ├── GAT_model.py          # GATv2 architecture implementation
│   └── parse_args.py         # Hyperparameter configuration
├── Feature_calculation/
│   ├── Feature_calculation.py # Molecular feature extraction
│   └── id_names.py           # Feature vocabulary definitions
├── Baseline_model/
│   └── baseline_models.py    # Comparative ML models
├── train/
│   ├── train.py              # Training loop with augmentation
│   └── plot.py               # Visualization utilities
├── Validate_External/
│   └── Validate_External.py  # External validation pipeline
├── config.py                 # Global configuration
├── setup_GAT.py              # Main entry point
├── setup_base.py             # Baseline model runner
├── run_shap_analysis.py      # SHAP feature analysis
├── paremeter_tuning_bayes.py # Bayesian optimization
├── requirements.txt          # Dependencies
└── dataset/
    └── data.csv              # Experimental data
```

---

## 9. References and Citation

### 9.1 Key Literature

1. **GATv2:** Brody, S., Alon, U., & Yahav, E. (2022). How Attentive are Graph Attention Networks? *ICLR 2022*.

2. **Original GAT:** Veličković, P., et al. (2018). Graph Attention Networks. *ICLR 2018*.

3. **Molecular GNNs:** Gilmer, J., et al. (2017). Neural Message Passing for Quantum Chemistry. *ICML 2017*.

4. **SHAP:** Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.

5. **Mordred Descriptors:** Moriwaki, H., et al. (2018). Mordred: a molecular descriptor calculator. *J. Cheminformatics*.

### 9.2 Citation

If you use this code in your research, please cite:

```bibtex
@software{gatchiral2024,
  author = {Weeqe1},
  title = {GATChiral: Graph Attention Networks for Chromatographic 
           Retention Time Prediction of Chiral Molecules},
  year = {2024},
  url = {https://github.com/Weeqe1/Enantioseparation},
  version = {1.0.0}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Bridging computational chemistry and deep learning for intelligent enantioseparation</i>
</p>
