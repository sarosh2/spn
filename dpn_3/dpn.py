import torch
import torch.nn as nn
import torch.nn.functional as F

class DPN(nn.Module):
    def __init__(self, input_features, total_nodes, output_nodes, minimal=True, hidden_dims=None):
        super().__init__()
        self.minimal = minimal
        self.output_nodes = output_nodes
        self.total_nodes = total_nodes
        self.hidden_nodes = total_nodes - output_nodes
        
        if hidden_dims is None:
            if self.minimal:
                hidden_dims = [self.hidden_nodes, self.output_nodes]
            else:
                hidden_dims = [1] * total_nodes
        
        self.layers = nn.ModuleList()
        in_size = input_features
        for out_size in hidden_dims:
            self.layers.append(nn.Linear(in_size, out_size))
            in_size += out_size

    def forward(self, x):
        if self.minimal:
            h1 = self.layers[0](x)
            # Optional: apply nonlinearity, e.g. ReLU
            h1 = F.relu(h1)
            # Concatenate input and h1 along the feature dimension
            concat = torch.cat([x, h1], dim=-1)
            return self.layers[1](concat)
        
        else:
            hidden_nodes = self.hidden_nodes
            node = 0
            for layer in self.layers:
                out = layer(x)
                if node < hidden_nodes:
                    out = F.gelu(out)
                x = torch.cat([x, out], dim=-1)
                node += layer.out_features

            return x[:, -self.output_nodes:]
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())