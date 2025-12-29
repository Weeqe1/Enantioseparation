# =============================================================================
# Graph Attention Network (GAT) Model Implementation
# =============================================================================
# This module implements a Graph Attention Network using GATv2Conv layers
# with multi-layer perceptron predictor head for graph-level prediction tasks.
# =============================================================================

# Standard library imports
# (None required for this module)

# Third-party library imports - PyTorch core
import torch
import torch.nn.functional as F

# Third-party library imports - PyTorch Geometric
from torch_geometric.nn import GATv2Conv, global_mean_pool

# Third-party library imports - PyTorch Neural Network modules
from torch.nn import BatchNorm1d, Linear, Sequential, ReLU, Dropout


class GAT(torch.nn.Module):
    """
    Graph Attention Network (GAT) model with GATv2 convolution layers.

    This implementation uses GATv2Conv for dynamic attention mechanism and
    includes residual connections, batch normalization, and an MLP predictor
    head for improved performance on graph-level prediction tasks.

    Architecture:
    - Multiple GATv2Conv layers with multi-head attention
    - Residual connections between layers
    - Batch normalization for stable training
    - Global mean pooling for graph-level representation
    - Multi-layer perceptron predictor head
    """

    def __init__(self, node_features, edge_features, hidden_dim, output_dim, heads, num_layers, dropout):
        """
        Initialize the GAT model with dynamic attention and MLP predictor head.

        Changes from original implementation:
        1. Replaced single linear layer with MLP predictor
        2. Added MLP with hidden layers and non-linear activations

        Args:
            node_features (int): Number of input features for each node.
            edge_features (int): Number of features for each edge.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of the output from GAT layers.
            heads (int): Number of attention heads in GAT layers.
            num_layers (int): Number of GAT convolutional layers.
            dropout (float): Dropout rate for regularization.
        """
        super(GAT, self).__init__()

        # Store hyperparameters
        self.num_layers = num_layers
        self.dropout = dropout

        # =================================================================
        # GAT Convolutional Layers
        # =================================================================
        # Create a list to hold the GAT convolutional layers
        self.convs = torch.nn.ModuleList()

        # First GAT layer: node_features -> hidden_dim * heads
        # Uses GATv2Conv for dynamic attention computation
        self.convs.append(GATv2Conv(node_features, hidden_dim, heads=heads, edge_dim=edge_features))

        # Intermediate GAT layers: (hidden_dim * heads) -> (hidden_dim * heads)
        # Each layer maintains the feature dimension after concatenating attention heads
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_features))

        # Final GAT layer: (hidden_dim * heads) -> output_dim
        # Uses single attention head to produce final node representations
        self.convs.append(GATv2Conv(hidden_dim * heads, output_dim, heads=1, edge_dim=edge_features))

        # =================================================================
        # Residual Connection Layers
        # =================================================================
        # Linear projections for residual connections to match dimensions
        self.residuals = torch.nn.ModuleList()

        # Residual connections for intermediate layers
        for _ in range(num_layers - 2):
            self.residuals.append(torch.nn.Linear(hidden_dim * heads, hidden_dim * heads))

        # Final residual connection
        self.residuals.append(torch.nn.Linear(hidden_dim * heads, output_dim))

        # =================================================================
        # Batch Normalization Layers
        # =================================================================
        # Batch normalization for stable training and faster convergence
        self.batch_norms = torch.nn.ModuleList()

        # Batch norm for intermediate layers
        for _ in range(num_layers - 1):
            self.batch_norms.append(BatchNorm1d(hidden_dim * heads))

        # Batch norm for final layer
        self.batch_norms.append(BatchNorm1d(output_dim))

        # =================================================================
        # Multi-Layer Perceptron Predictor Head
        # =================================================================
        # Replaces simple linear layer with sophisticated MLP for better prediction capability
        self.predictor = Sequential(
            Linear(output_dim, hidden_dim),  # Expand feature dimensions
            ReLU(),  # Non-linear activation
            Dropout(dropout),  # Prevent overfitting
            Linear(hidden_dim, hidden_dim // 2),  # Compress features
            ReLU(),  # Non-linear activation
            Dropout(dropout * 0.8),  # Weaker dropout
            Linear(hidden_dim // 2, 1)  # Output layer
        )

    def forward(self, data):
        """
        Forward pass through the GAT model with dynamic attention and MLP predictor.

        The forward pass consists of:
        1. Node-level processing through multiple GAT layers
        2. Residual connections and batch normalization
        3. Global pooling for graph-level representation
        4. MLP predictor for final output

        Changes from original implementation:
        - Replaced simple linear output with MLP predictor

        Args:
            data: A PyTorch Geometric data object containing:
                - x (torch.Tensor): Node features [num_nodes, node_features]
                - edge_index (torch.Tensor): Edge connectivity [2, num_edges]
                - edge_attr (torch.Tensor): Edge features [num_edges, edge_features]
                - batch (torch.Tensor): Batch assignment for nodes

        Returns:
            torch.Tensor: The output predictions for each graph in the batch [batch_size].
        """
        # Extract graph components from data object
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        residual = None

        # =================================================================
        # Node-level Processing through GAT Layers
        # =================================================================
        # Process the graph through each GAT layer with residual connections
        for i, conv in enumerate(self.convs):
            # Store input for residual connection (skip first layer)
            if i > 0:
                residual = x

            # Apply GAT convolution with dynamic attention
            x = conv(x, edge_index, edge_attr)

            # Apply batch normalization for stable training
            x = self.batch_norms[i](x)

            # Apply ELU activation function (smooth, non-zero gradient)
            x = F.elu(x)

            # Apply dropout for regularization
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Add residual connection if not the first layer
            if residual is not None:
                x += self.residuals[i - 1](residual)

        # =================================================================
        # Graph-level Aggregation
        # =================================================================
        # Global mean pooling to get graph-level representation from node features
        # Transforms [total_nodes_in_batch, output_dim] -> [batch_size, output_dim]
        x = global_mean_pool(x, data.batch)

        # =================================================================
        # Final Prediction
        # =================================================================
        # Pass through MLP predictor instead of simple linear layer
        # This provides more sophisticated feature transformation for final prediction
        x = self.predictor(x)

        # Reshape the output to a 1D tensor [batch_size]
        return x.view(-1)