import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_channel, out_channel):
    """Return a block of conv2d-bn-relu-maxpooling"""
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )
    

class EmbeddingNet(nn.Module):
    """Define the Matching network"""
    def __init__(self, in_channel, out_channel, num_hiddens, input_shape,  num_layers=4) -> None:
        super().__init__()
        assert num_layers > 2
        self.input_shape = input_shape
        
        self.embedding = []
        self.embedding += conv_block(in_channel, num_hiddens)
        for _ in range(num_layers - 2):
            self.embedding += conv_block(num_hiddens, num_hiddens)
        self.embedding += conv_block(num_hiddens, out_channel)
        self.embedding = nn.Sequential(*self.embedding)
        
        w, d = self.__compute_map_size()
        self._out_size = out_channel * w * d
        
    @property
    def out_size(self):
        return self._out_size
        
    def __compute_map_size(self):
        """compute the width and heigth of final feature map"""
        width, height = self.input_shape
        for layer in self.embedding:
            if isinstance(layer, nn.Conv2d):
                k, s, p = layer.kernel_size, layer.stride, layer.padding
                width = (width - k[0] + 2 * p[0]) // s[0] + 1   # tuple for Con2d
                height = (height - k[1] + 2 * p[1]) // s[1] + 1
            elif isinstance(layer, nn.MaxPool2d):
                k, s, p = layer.kernel_size, layer.stride, layer.padding
                width = (width - k + 2 * p) // s + 1            # int for MaxPool2d
                height = (height - k + 2 * p) // s + 1
        return width, height
        
    def forward(self, X):
        return self.embedding(X).view(X.size(0), -1)
    
    
class AttentionClassifier(nn.Module):
    """Define the classifier with softmax attention"""
    def __init__(self) -> None:
        super().__init__()
     
    def forward(self, similarities, ys):
        """
        Parameters:
            similarities (Tensor): a tensor represents the similarities between support set and query set, 
                                size of (n_ways*num_query, n_ways*num_support)
            ys (Tensor): a tensor consisted of labels of support set, size of (n_ways*num_support, num_classes)
        """
        score = F.softmax(similarities, dim=1)
        preds = score @ ys   # shape: (batch_size, num_classes)
        return preds
    
    
class BidirectionLSTMEmbedding(nn.Module):
    """Define a bidirectional LSTM for full context embedding (FCE)"""
    def __init__(self, input_size, layer_size, devicce, batch_first=True) -> None:
        super().__init__()
        self.num_layers = len(layer_size)
        self.num_hiddens = layer_size[0]
        self.device = devicce
        self.fce = nn.LSTM(
            input_size=input_size, hidden_size=self.num_hiddens, num_layers=self.num_layers,
            batch_first=batch_first, bidirectional=True
        )
        
    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(0)  # add batch axis
        h0 = torch.randn(size=(self.num_layers * 2, X.size(0), self.num_hiddens)).to(self.device)
        c0 = torch.randn(size=(self.num_layers * 2, X.size(0), self.num_hiddens)).to(self.device)
        output, (_, _) = self.fce(X, (h0, c0))
        output = output.squeeze()
        return output.view(output.size(0), -1)
    
