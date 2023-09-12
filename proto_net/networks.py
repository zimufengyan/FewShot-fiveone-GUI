import torch
from torch import nn


def conv_block(in_channel, out_channel):
    """Return a block of conv2d-bn-relu-maxpooling"""
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )
    

class ProtoEmbeddingNet(nn.Module):
    """Define the Prototypical network"""
    def __init__(self, in_channel, out_channel, num_hiddens, num_layers=4) -> None:
        super().__init__()
        assert num_layers > 2
        self.embedding = []
        self.embedding += conv_block(in_channel, num_hiddens)
        for _ in range(num_layers - 2):
            self.embedding += conv_block(num_hiddens, num_hiddens)
        self.embedding += conv_block(num_hiddens, out_channel)
        self.embedding = nn.Sequential(*self.embedding)
        
    def forward(self, X):
        return self.embedding(X).view(X.size(0), -1)