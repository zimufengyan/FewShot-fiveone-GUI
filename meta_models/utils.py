import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    
    
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def get_scheduler(optimizer, policy, **kwargs):
    """
    Return a learning rate scheduler
    Params:
        optimizer : the optimizer of network or model
        policy (str) : speci fies the scheduling policy of the sheduler: 'Step' | 'Cosine' | 'Pleteau'
    """
    if policy.lower() == 'step':
        lr_decay_iters = kwargs.get('lr_decay_iters', 50)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=gamma)
    elif policy.lower() == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif policy.lower() == 'cosine':
        n_epochs = kwargs.get('n_epochs', 100)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return ValueError(f"the param of 'policy' should be one of 'step', 'cosine', or 'pleteau, rather than {policy}.")
    return scheduler


def compute_accuracy(y_hat, y):
    """Compute the percent of correct predictions.`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    b_size = y.size(0)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum().item()) / b_size


def compute_map_size(nets, width, height):
    """compute the width and heigth of final feature map"""       
    def compute(net, w, h):
        if isinstance(net, nn.Sequential) or isinstance(net, list):
            for m in net:
                w, h = compute(m, w, h)
        elif isinstance(net, nn.Conv2d):
            k, s, p = net.kernel_size, net.stride, net.padding
            w = (w - k[0] + 2 * p[0]) // s[0] + 1   # tuple for Con2d
            h = (h - k[1] + 2 * p[1]) // s[1] + 1
        elif isinstance(net, nn.MaxPool2d):
            k, s, p = net.kernel_size, net.stride, net.padding
            w = (w - k + 2 * p) // s + 1            # int for MaxPool2d
            h = (h - k + 2 * p) // s + 1
        return w, h
    
    return compute(nets, width, height)


class ValueMeter:
    """Define a class for recording a certain variable's average, sum and count"""
    def __init__(self) -> None:
        self.reset()
        
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        
    def update(self, value, n=1):
        self.sum += value * n
        self.cnt += n
        self.avg = self.sum / self.cnt
            