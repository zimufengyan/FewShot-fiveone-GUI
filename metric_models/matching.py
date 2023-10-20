import os
import torch 
from torch import nn 
import torch.nn.functional as F
import itertools
import sys

from metric_models.utils import *
from metric_models.networks import EmbeddingNet, BidirectionLSTMEmbedding
from metric_models.base_model import MetricModelBase


class MatchingModel(MetricModelBase):
    """Define the matching model class"""
    def __init__(self, input_shape, out_channels, num_hiddens, lr=0.001, gpu_ids=[0], 
                 use_fce=True, num_fce_hiddens=32, num_fce_layers=1,
                 distance='cosine', is_train=True, **kwargs) -> None:
        super().__init__(gpu_ids, distance=distance, is_train=is_train, **kwargs)
        
        self.use_fce = use_fce
        
        self.embedding_net = EmbeddingNet(
            input_shape=input_shape, out_channels=out_channels, num_hiddens=num_hiddens, num_layers=4, 
        )
        
        self.model_name = 'MatchingNet'
        self.nets.append('embedding')
        
        if use_fce:
            out_size = self.embedding_net.out_size
            input_size = out_size[0] * out_size[1] * out_size[2]
            self.fce_net = BidirectionLSTMEmbedding(
                input_size=input_size, layer_size=[num_fce_hiddens]*num_fce_layers, 
                device=self.device, batch_first=True
            )
            self.nets.append('fce')
        
        
        if is_train:
            if use_fce:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.embedding_net.parameters(), self.fce_net.parameters()), 
                    lr=lr
                )
            else:
                self.optimizer = torch.optim.Adam(
                    self.embedding_net.parameters(), lr=lr
                )
     
    def init_net(self, init_type='normal', init_gian=0.2):
        """Initialize the weights of embedding net"""
        init_weights(self.embedding_net, init_type=init_type, init_gain=init_gian)     
        
    def _compute_distance(self, x, y):
        """Conpute the distance between x and y"""
        if x.size(2) != y.size(2):
            raise RuntimeError(f"The size of tensor x ({x.size(2)}) must match the size of tensor y ({y.size(2)}) at dimension 2")
        n, m = x.size(1), y.size(1)
        k, d = x.size(0), x.size(2)   
        
        x = x.unsqueeze(2).expand(k, n, m, d)
        y = y.unsqueeze(1).expand(k, n, m, d)
        
        if self.distance == 'euclidean':
            return torch.pow((x - y), 2).sum(3)
        elif self.distance == 'cosine':
            return 1 - F.cosine_similarity(x, y, dim=3)
        else:
            raise ValueError(f"unsported distance function ({self.distance})")  
        
    def _forward(self, Xs, Xq, n_ways):
        """
        The forward function of Matching Model
        num_support should be 1 for matching network (N-way One-shot)
        """
        self.embedding_net.train()
        num_support = Xs.size(0) // n_ways
        num_query = Xq.size(0) // n_ways
        if num_support != 1:
            raise RuntimeError("Matching network only support one-shot task, so the number of samples in suport set must be one")
        
        X = torch.cat([Xs, Xq], dim=0)
        X = X.to(self.device)
        
        Z = self.embedding_net(X)
        Z = Z.view(Z.size(0), -1)
        
        # Zs shape: (n_ways*num_support, c*w*h), Zq shape: (n-ways*num_query, c*w*h), num_support == 1
        # cats shape: (n_ways*num_query, n_ways*num_support + 1, c*w*h)
        Zq, Zs = Z[n_ways*num_support:], Z[:n_ways*num_support]     
        cats = [torch.cat([Zs, Zq[i].unsqueeze(0)], dim=0) for i in range(n_ways*num_query)]
        cats = torch.stack(cats)
        
        if self.use_fce:
            # cats shape: (n_ways*num_query, n_ways*num_support + 1, num_fce_hiddens)
            cats = self.fce_net(cats)
        
        # compute the (cosine) similarities between samples of query_set and support set
        similarities = 1 - self._compute_distance(cats[:, n_ways*num_support:], cats[:, :n_ways*num_support])
        # similarities shape: (n_ways*num_query, 1, n_ways*num_support) => (n_ways*num_query, n_ways*num_support)
        similarities = similarities.contiguous().view(n_ways * num_query, -1)
        # distance = self._compute_distance(Z[n_ways*num_support:], Z[:n_ways*num_support])

        # compute prediction probabilities
        target_inds = torch.arange(0, n_ways).view(n_ways, 1).expand(n_ways, num_query)
        target_inds = target_inds.contiguous().view(n_ways*num_query).long().to(self.device)
        target_one_hot = F.one_hot(target_inds).float()
        # log_p_y = F.log_softmax(-distance, dim=1).view(n_ways, num_query, -1)    
        p_y = F.softmax(similarities, dim=1)
        
        # compute loss and accuracy
        # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        loss_val = F.cross_entropy(p_y, target_one_hot)
        # _, y_hat = log_p_y.max(2)
        _, y_hat = p_y.max(1)
        acc_val = y_hat.eq(target_inds).float().mean()
        
        return loss_val, acc_val
    
    def pred(self, Xs, Xq, n_ways):
        num_support = Xs.size(0) // n_ways
        num_query = Xq.size(0)
        assert num_support == 1
        with torch.no_grad():
            X = torch.cat([Xs, Xq], dim=0)
            X = X.to(self.device)
            Z = self.embedding_net(X)
            Z = Z.view(Z.size(0), -1)
            
            Zq, Zs = Z[n_ways*num_support:], Z[:n_ways*num_support]     
            cats = [torch.cat([Zs, Zq[i].unsqueeze(0)], dim=0) for i in range(num_query)]
            cats = torch.stack(cats)
            if self.use_fce:
                cats = self.fce_net(cats)   
            # compute the distance between samples of query_set and support set
            similarities = 1 - self._compute_distance(cats[:, n_ways*num_support:], cats[:, :n_ways*num_support])
            similarities = similarities.contiguous().view(num_query, -1)
            p_y = F.softmax(similarities, dim=1)
            print(p_y.cpu())
            _, y_hat = p_y.max(1)
        
        return y_hat.cpu()