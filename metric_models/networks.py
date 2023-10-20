import torch
from torch import nn
import torch.nn.functional as F
import sys
import os

from metric_models.utils import *


def conv_block(in_channel, out_channel, with_pooling=True):
    """Return a block of conv2d-bn-relu-maxpooling"""
    nets = [
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    ]
    if with_pooling:
        nets += [nn.MaxPool2d(2)]
    return nn.Sequential(*nets)
    

class EmbeddingNet(nn.Module):
    """Define the embedding network"""
    def __init__(self, input_shape, out_channels=64, num_hiddens=64, num_layers=4, with_poolings=[True]*4) -> None:
        super().__init__()
        assert num_layers > 2
        assert num_layers == len(with_poolings)
        in_channel, w, h = input_shape
        
        self.embedding = []
        self.embedding += [conv_block(in_channel, num_hiddens, with_pooling=with_poolings[0])]
        for i in range(num_layers - 2):
            self.embedding += [conv_block(num_hiddens, num_hiddens, with_pooling=with_poolings[i+1])]
        self.embedding += [conv_block(num_hiddens, out_channels, with_pooling=with_poolings[-1])]
        self.embedding = nn.Sequential(*self.embedding)
        
        w, h = compute_map_size(self.embedding, w, h)
        self._out_size = (out_channels, w, h)
        
    @property
    def out_size(self):
        return self._out_size
        
    def forward(self, X):
        return self.embedding(X)
    

class DistanceNet(nn.Module):
    """Define the distance network for computing relation score using nerual networks"""
    def __init__(self, input_shape, num_hiddens=64, with_poolings=[True]*2) -> None:
        super().__init__()
        assert len(with_poolings) == 2
        
        in_channels, width, height = input_shape
        
        nets = [
            conv_block(in_channels, num_hiddens, with_pooling=with_poolings[0]), 
            conv_block(num_hiddens, num_hiddens, with_pooling=with_poolings[1])
        ]
        
        width, height = compute_map_size(nets, width, height)
        H = num_hiddens * width * height
        
        nets += [
            nn.Flatten(),
            nn.Linear(H, 8), nn.ReLU(inplace=True),
            nn.Linear(8, 1), nn.Sigmoid()
        ]
        
        self.net = nn.Sequential(*nets)
        
    def forward(self, X):
        """X: the concat result of support set and query set along the feature dim"""
        return self.net(X)  # shape: (X.size(0), 1)
    

class SelfAttention(nn.Module):
    """
    Define a self-attention module for Self-Attention Relation Net (SARN).
    Implemented by of embedding Gaussian version with bottleneck of 'in_channels // 2' channels
    reference from Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He, "Non-local neural networks," in CVPR, 2018
    """
    def __init__(self, in_channels) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.out_conv = nn.Conv2d(in_channels // 2, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, X):
        """
        The forward function of self-attenion
        Parameters:
            X (Tensor): input feature map, shape of (batch_size, c_in, w, h)
        Return:
            out (Tensor): self-attenion value + input feature, shape of (batch_size, c, w, h)
        """
        b_size, c, w, h = X.size()
        n = w * h
        query = self.query_conv(X).view(b_size, -1, n).permute(0, 2, 1)    # Q shape : (b_size, n, c // 2)  
        key = self.key_conv(X).view(b_size, -1, n)          # K.T shape: (b_size, c // 2, n)
        value = self.value_conv(X).view(b_size, -1, n)      # V shape: (b_size, c // 2, n)
        score = torch.bmm(query, key)           # Q @ K.T
        score = F.softmax(score, dim=1)        # shape: (b_size, n, n)
        
        out = torch.bmm(value, score)
        out = out.view(b_size, c // 2, w, h)           # shape: (b_size, c // 2, w, h)
        out = self.out_conv(out)                       # shape: (b_size, c, w, h)
        
        return self.gamma * out + X
    
    
class BidirectionLSTMEmbedding(nn.Module):
    """Define a bidirectional LSTM for full context embedding (FCE)"""
    def __init__(self, input_size, layer_size, device, batch_first=True) -> None:
        super().__init__()
        self.num_layers = len(layer_size)
        self.num_hiddens = layer_size[0]
        self.device = device
        
        self.fce = nn.LSTM(
            input_size=input_size, hidden_size=self.num_hiddens, num_layers=self.num_layers,
            batch_first=batch_first, bidirectional=True
        )
        
    def forward(self, X: torch.Tensor):
        h0 = torch.randn(size=(self.num_layers * 2, X.size(0), self.num_hiddens)).to(self.device)
        c0 = torch.randn(size=(self.num_layers * 2, X.size(0), self.num_hiddens)).to(self.device)
        output, (_, _) = self.fce(X, (h0, c0))
        return output
    
    