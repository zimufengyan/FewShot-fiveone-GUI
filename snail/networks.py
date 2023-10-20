import math
import numpy as np
import torch
from torch import nn 
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """A 1D causal convolution layer.

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions per step, and T is the number of steps.
    Output: (B, D_out, T), where B is the minibatch size, D_out is the number
        of dimensions in the output, and T is the number of steps.

    Arguments:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride=1, bias=True):
        super().__init__()
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=self.padding, dilation=dilation, bias=bias,
        )
        
    def forward(self, X):
        return self.conv(X)[:, :, :-self.dilation]
    
    
class DenseBlock(nn.Module):
    """Two parallel 1D causal convolution layers w/tanh and sigmoid activations

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.

    Arguments:
        in_channels (int): number of input channels
        filters (int): number of filters
    """
    def __init__(self, in_channels, filters, dilation, 
                 kernel_size=2, stride=1, bias=True):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, filters, kernel_size, dilation, stride, bias)
        self.conv2 = CausalConv1d(in_channels, filters, kernel_size, dilation, stride, bias)
        
    def forward(self, X):
        # with residule connect
        xf = self.conv1(X)
        xg = self.conv2(X)
        activations = torch.tanh(xf) * torch.sigmoid(xg)
        return torch.cat([X, activations], dim=1)
    

class TCBlock(nn.Module):
    """A stack of DenseBlocks which dilates to desired sequence length

    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in + L * F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `L` is the number of layers, 
        `F` is the number of filters, and `T` is the length of the input sequence.

    Arguments:
        in_channels (int): channels for the input
        seq_length (int): length of the sequence. The number of denseblock layers
            is log base 2 of `seq_length`.
        filters (int): number of filters per channel
    """
    def __init__(self, in_channels, seq_length, filters, bias=True):
        super().__init__()
        blks = []
        self.in_channels = in_channels
        self.filters = filters
        self.layers = int(np.ceil(math.log(seq_length, 2)))
        for i in range(self.layers):
            blks.append(DenseBlock(in_channels + i * filters, filters, dilation=int(2**i), bias=bias))
        self.dense_blocks = nn.Sequential(*blks)
        
    @property
    def out_channels(self):
        return self.filters * self.layers + self.in_channels
    
    def forward(self, X):
        return self.dense_blocks(X)
    
    
class AttentionBlock(nn.Module):
    """An attention mechanism similar to Vaswani et al (2017)

    Input: (B, T, D), where `B` is the input minibatch size, 
    `D` is the dimensions of each feature, `T` is the length of
    the sequence.

    Output: (B, T, D + v), where `V` is the size of the attention values.

    Arguments:
        in_channels (int): the number of dimensions (or channels) of each element
            in the input sequence
        num_key (int): the size of the attention keys
        num_query (int): the size of the attention values
    """
    def __init__(self, in_channels, num_key, num_value):
        super().__init__()
        self.in_channels = in_channels
        self.num_key = num_key
        self.num_value = num_value
        self.key_layer = nn.Linear(in_channels, num_key)
        self.value_layer = nn.Linear(in_channels, num_value)
        self.query_layer = nn.Linear(in_channels, num_key)
        self.sqrt_k = math.sqrt(num_key)
        
    @property
    def out_channels(self):
        return self.in_channels + self.num_value
        
    def forward(self, X):
        mask = torch.ByteTensor([[1 if i>j else 0 for i in range(X.size(1))] for j in range(X.size(1))]).to(X.device)
        key = self.key_layer(X)     # size: (B, T, num_key)
        query = self.query_layer(X) # size: (B, T, num_key)
        value = self.value_layer(X) # size: (B, T, num_value)
        score = torch.bmm(query, key.permute(0, 2, 1))  # size: (B, T, T)
        score.data.masked_fill_(mask, -float('inf'))
        score = F.softmax(score / self.sqrt_k, dim=1)
        out = torch.bmm(score, value)    # size: (B, T, num_value)
        return torch.cat([X, out], dim=2)   # size: (B, T, num_value + D)
    
    
class ConvBlock(nn.Module):
    """Define a conv2d block, containing conv2d-bn-relu-maxpooling"""
    def __init__(self, in_channels, num_hiddens=64, bias=True) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, num_hiddens, 3, 1, 1, bias=bias)
        self.bn = nn.BatchNorm2d(num_hiddens,  momentum=1.,
            track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, X):
        X = self.conv2d(X)
        X = self.bn(X)
        self.relu(X)
        return self.pool(X)
    
    def __iter__(self):
        nets = [self.conv2d, self.bn, self.relu, self.pool]
        return iter(nets)


class OmniglotEmbedding(nn.Module):
    """Define the embedding network"""
    def __init__(self, in_channels=1, out_channels=64, num_hiddens=64, bias=True) -> None:
        super().__init__()
    
        self.blk_1 = ConvBlock(in_channels, num_hiddens, bias=bias)
        self.blk_2 = ConvBlock(num_hiddens, num_hiddens, bias=bias)
        self.blk_3 = ConvBlock(num_hiddens, num_hiddens, bias=bias)
        self.blk_4 = ConvBlock(num_hiddens, out_channels, bias=bias)

    def forward(self, X):
        out = self.blk_1(X)
        out = self.blk_2(out)
        out = self.blk_3(out)
        out = self.blk_4(out)
        return out.view(X.size(0), -1)
        
    
class ResNet(nn.Module):
    """A simple residual network"""
    def __init__(self, in_channels, out_channels, dropout=0.9, bias=True):
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
        
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X):
        out = self.embedding(X)
        out += self.conv_1x1(X) 
        return self.dropout(self.pool(out))


class MiniImageEmbedding(nn.Module):
    def __init__(self, in_channels=3, dropout=0.9, bias=True):
        resnet_blks = []
        self.resnet_1 = ResNet(in_channels, 64, dropout=dropout, bias=bias)
        self.resnet_2 = ResNet(in_channels, 96, dropout=dropout, bias=bias)
        self.resnet_3 = ResNet(in_channels, 128, dropout=dropout, bias=bias)
        self.resnet_4 = ResNet(in_channels, 256, dropout=dropout, bias=bias)
        self.conv_1 = nn.Conv2d(256, 2048, 1, 1, 0, bias=bias)
        self.pool = nn.AveragePool(6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv2d(2048, 384, 1, 1, 0)
        
    def forward(self, X):
        out = self.resnet_1(X)
        out = self.resnet_2(out)
        out = self.resnet_3(out)
        out = self.resnet_4(out)
        out = self.conv_1(out)
        out = self.pool(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv_2(out)
        
        return out.view(X.size(0), -1)