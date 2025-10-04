import torch
from torch_geometric.data import Data
from src.models.gnn import CGCNN

def test_gnn_forward():
    # Dummy graph
    x = torch.rand(4, 3)
    edge_index = torch.tensor([[0,1,2,3],[1,2,3,0]], dtype=torch.long)
    edge_attr = torch.rand(4, 32)
    batch = torch.zeros(4, dtype=torch.long)
    y = torch.rand(4, 1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=y)
    model = CGCNN()
    out = model(data)
    assert out.shape[1] == 1
