"""
Graph Neural Network Encoder for Traffic Signal Control.

This module implements a GNN that encodes the spatial relationships between
traffic signals/intersections, enabling the SAC agent to learn coordinated
control policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphConvLayer(nn.Module):
    """
    Simple Graph Convolution Layer (simplified GraphSAGE).
    
    Aggregates neighbor features and combines with self features.
    """
    
    def __init__(self, in_features: int, out_features: int, aggregator: str = "mean"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        # Linear transformation for self features
        self.linear_self = nn.Linear(in_features, out_features)
        # Linear transformation for aggregated neighbor features
        self.linear_neigh = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges], where edge_index[0] are source
                       nodes and edge_index[1] are target nodes
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        
        # Self transformation
        self_features = self.linear_self(x)
        
        # Neighbor aggregation
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Gather source node features
        source_features = x[source_nodes]  # [num_edges, in_features]
        
        # Aggregate by target node (mean aggregation)
        # Initialize aggregated features
        aggregated = torch.zeros(num_nodes, self.in_features, device=x.device)
        counts = torch.zeros(num_nodes, 1, device=x.device)
        
        # Scatter add source features to target nodes
        aggregated = aggregated.scatter_add(0, target_nodes.unsqueeze(1).expand(-1, self.in_features), source_features)
        counts = counts.scatter_add(0, target_nodes.unsqueeze(1), torch.ones(len(target_nodes), 1, device=x.device))
        
        # Mean aggregation
        counts = counts.clamp(min=1)  # Avoid division by zero
        aggregated = aggregated / counts
        
        # Transform aggregated neighbor features
        neigh_features = self.linear_neigh(aggregated)
        
        # Combine self and neighbor features
        out = self_features + neigh_features
        
        return out


class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder for traffic signal state.
    
    Encodes the state of multiple intersections considering their spatial
    relationships in a graph structure.
    """
    
    def __init__(
        self,
        node_features: int = 4,      # Features per intersection node
        hidden_dim: int = 64,        # Hidden dimension
        output_dim: int = 64,        # Output embedding dimension
        num_layers: int = 2,         # Number of GNN layers
        dropout: float = 0.1         # Dropout rate
    ):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.gnn_layers.append(GraphConvLayer(in_dim, out_dim))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [batch_size, num_nodes, node_features] or 
               [num_nodes, node_features] for single graph
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            Node embeddings [batch_size, num_nodes, output_dim] or
            [num_nodes, output_dim] for single graph
        """
        # Handle batched vs single input
        is_batched = x.dim() == 3
        if is_batched:
            batch_size, num_nodes, _ = x.shape
            # Process each graph in batch (simple approach for small graphs)
            outputs = []
            for i in range(batch_size):
                out = self._forward_single(x[i], edge_index)
                outputs.append(out)
            return torch.stack(outputs, dim=0)
        else:
            return self._forward_single(x, edge_index)
    
    def _forward_single(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Process a single graph."""
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers with residual connections
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            residual = x
            x = gnn_layer(x, edge_index)
            x = layer_norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
                # Residual connection
                if residual.shape == x.shape:
                    x = x + residual
            
        return x


class TrafficGNNEncoder(nn.Module):
    """
    Traffic-specific GNN encoder that handles the 4-intersection setup.
    
    Converts the 16-dimensional observation into graph format and encodes it.
    """
    
    def __init__(
        self,
        obs_dim: int = 16,           # Input observation dimension
        hidden_dim: int = 64,        # GNN hidden dimension
        output_dim: int = 64,        # Output dimension per node
        num_gnn_layers: int = 2      # Number of GNN layers
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.num_signals = 4
        self.features_per_signal = 4  # 3 lanes + 1 signal state per direction
        
        # GNN encoder
        self.gnn = GNNEncoder(
            node_features=self.features_per_signal,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_gnn_layers
        )
        
        # Edge index for 4-intersection ring topology
        # Each signal is connected to adjacent signals (circular)
        # 0 <-> 1 <-> 2 <-> 3 <-> 0
        self.register_buffer(
            'edge_index',
            torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 0],  # Source nodes
                [1, 0, 2, 1, 3, 2, 0, 3]   # Target nodes
            ], dtype=torch.long)
        )
        
        # Final output dimension: num_signals * output_dim
        self.output_dim = self.num_signals * output_dim
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor [batch_size, 16] or [16]
        
        Returns:
            Encoded features [batch_size, output_dim] or [output_dim]
        """
        is_batched = obs.dim() == 2
        if not is_batched:
            obs = obs.unsqueeze(0)
        
        batch_size = obs.shape[0]
        
        # Reshape observation to graph format
        # obs[:, 0:12] = vehicle counts (4 directions × 3 lanes)
        # obs[:, 12:16] = signal states (4 signals)
        
        # Each node (signal) gets: 3 lane counts + 1 signal state
        node_features = torch.zeros(batch_size, self.num_signals, self.features_per_signal, device=obs.device)
        
        for i in range(self.num_signals):
            # Lane counts for this direction (3 lanes)
            node_features[:, i, :3] = obs[:, i*3:(i+1)*3]
            # Signal state
            node_features[:, i, 3] = obs[:, 12 + i]
        
        # Apply GNN
        node_embeddings = self.gnn(node_features, self.edge_index)  # [batch, 4, output_dim]
        
        # Flatten node embeddings
        output = node_embeddings.view(batch_size, -1)  # [batch, 4 * output_dim]
        
        if not is_batched:
            output = output.squeeze(0)
        
        return output


# Quick test
if __name__ == "__main__":
    print("Testing GNN Encoder...")
    
    # Create encoder
    encoder = TrafficGNNEncoder(obs_dim=16, hidden_dim=64, output_dim=64)
    print(f"Encoder output dim: {encoder.output_dim}")
    
    # Test with single observation
    obs = torch.randn(16)
    out = encoder(obs)
    print(f"Single input shape: {obs.shape} -> Output shape: {out.shape}")
    
    # Test with batched observations
    obs_batch = torch.randn(32, 16)
    out_batch = encoder(obs_batch)
    print(f"Batch input shape: {obs_batch.shape} -> Output shape: {out_batch.shape}")
    
    # Test GNN component directly
    gnn = GNNEncoder(node_features=4, hidden_dim=64, output_dim=64)
    x = torch.randn(4, 4)  # 4 nodes, 4 features each
    edge_index = torch.tensor([[0,1,1,2,2,3,3,0],[1,0,2,1,3,2,0,3]])
    gnn_out = gnn(x, edge_index)
    print(f"GNN direct test: {x.shape} -> {gnn_out.shape}")
    
    print("All tests passed!")
