import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=64):
        super(GraphSAGE, self).__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):

        # First graph convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # Second graph convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Fraud classification
        x = self.classifier(x)

        return x.squeeze(-1)