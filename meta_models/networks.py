import torch
from torch import nn 
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Define a conv2d block, containing conv2d-bn-relu-maxpooling"""
    def __init__(self, in_channels, num_hiddens=64) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, num_hiddens, 3, 1, 1)
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


def conv_block_functioin(X, w_conv, b_conv, w_bn, b_bn):
    """Define a functional conv block"""
    X = F.conv2d(X, w_conv, b_conv, stride=1, padding=1)
    X = F.batch_norm(
        X, momentum=1.0, 
        weight=w_bn, bias=b_bn, 
        running_mean=None, running_var=None,
        training=True
    )
    F.relu(X, inplace=True)
    return F.max_pool2d(X, kernel_size=2, stride=2)


class Classifier(nn.Module):
    """Define a classifier based convs and fcns for model"""
    def __init__(self, input_shape, n_classes, num_hiddens=64) -> None:
        super().__init__()
        self.input_shape = input_shape
        in_channels, w, h = input_shape
        self.blk_1 = ConvBlock(in_channels, num_hiddens)
        self.blk_2 = ConvBlock(num_hiddens, num_hiddens)
        self.blk_3 = ConvBlock(num_hiddens, num_hiddens)
        self.blk_4 = ConvBlock(num_hiddens, num_hiddens)
        
        w, h = self._compute_map_size()
        self.fcn = nn.Linear(num_hiddens * w * h, n_classes)
        
    def forward(self, X):
        X = self.blk_1(X)
        X = self.blk_2(X)
        X = self.blk_3(X)
        X = self.blk_4(X).view(X.size(0), -1)
        X = self.fcn(X)

        return X
    
    def _compute_map_size(self):
        _, w, h = self.input_shape
        for _ in range(4):
            w = (w - 2) // 2 + 1
            h = (h - 2) // 2 + 1
        return w, h
            
    @staticmethod
    def functional_forward(X, params):
        """Define functional forward for inner loop to keep original parameters of meta-learner"""
        X = conv_block_functioin(
            X, params['blk_1.conv2d.weight'], params['blk_1.conv2d.bias'],
            params['blk_1.bn.weight'], params['blk_1.bn.bias'],
        )
        X = conv_block_functioin(
            X, params['blk_2.conv2d.weight'], params['blk_2.conv2d.bias'],
            params['blk_2.bn.weight'], params['blk_2.bn.bias'],
        )
        X = conv_block_functioin(
            X, params['blk_3.conv2d.weight'], params['blk_3.conv2d.bias'],
            params['blk_3.bn.weight'], params['blk_3.bn.bias'],
        )
        X = conv_block_functioin(
            X, params['blk_4.conv2d.weight'], params['blk_4.conv2d.bias'],
            params['blk_4.bn.weight'], params['blk_4.bn.bias'],
        )
        
        X = X.view(X.size(0), -1)
        X = F.linear(X, params['fcn.weight'], params['fcn.bias'])
        
        return X
    
    
if __name__ == '__main__':
    model = Classifier((1, 28, 28), 5, 64)
    for name, param in model.named_parameters():
        print(name, param.size())