import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool

class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network baseline.
    """
    def __init__(self, node_dim=3, edge_dim=32, hidden_dim=128, out_dim=3, num_layers=3):
        super().__init__()
        self.conv1 = CGConv(node_dim, edge_dim, aggr='add')
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([
            CGConv(hidden_dim, edge_dim, aggr='add') for _ in range(num_layers - 1)
        ])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)  # Output vector for multi-target prediction

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.node_proj(x))
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out
