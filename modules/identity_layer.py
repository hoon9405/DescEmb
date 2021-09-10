import torch.nn as nn

class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, source):
        return source