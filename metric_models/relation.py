import os
import torch 
from torch import nn 
import itertools
import torch.nn.functional as F

from metric_models.nets import EmbeddingNet, DistanceNet
from metric_models.base_model import MetricModelBase


class RelationModel(MetricModelBase):
    """Define the matching model class"""
    def __init__(self, input_shape, num_hiddens=64, 
                 lr=0.001, gpu_ids=[0], is_train=True, **kwargs) -> None:
        super().__init__(gpu_ids, is_train=is_train, **kwargs)
        
        with_poolings = [True, True, False, False, True, True]
        self.embedding_net = EmbeddingNet(
            input_shape=input_shape, out_channels=num_hiddens, num_hiddens=num_hiddens, 
            num_layers=4, with_poolings=with_poolings[:4]
        )
        
        # input of distance net is the concat result of support set and query set along feature dim
        out_size = self.embedding_net.out_size
        d_input_shape = (out_size[0] * 2, out_size[1], out_size[2])   
        self.distance_net = DistanceNet(
            input_shape=d_input_shape, num_hiddens=num_hiddens, with_poolings=with_poolings[4:]    
        )
        
        self.model_name = 'RelationNet'
        self.nets = ['embedding', 'distance']      # store network name
        
        if is_train:
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.embedding_net.parameters(), self.distance_net.parameters()), lr=lr
            )
        
    def _forward(self, Xs, Xq, n_ways):
        """
        The forward function of Relation Model
        """
        num_support = Xs.size(0) // n_ways
        num_query = Xq.size(0) // n_ways
        
        X = torch.cat([Xs, Xq], dim=0)
        X = X.to(self.device)           # shape : (n_ways * (num_support + num-query), c_1, w_1, h_1 )
        
        Z = self.embedding_net(X)       # shape : (n_ways * (num_support + num-query), num_hiddens, w, h )
        size = list(Z.size())
        
        # compute the prototypical center point for each class of support set, sum for relation net
        proto_centers = Z[:n_ways*num_support].view(n_ways, num_support, *size[1:]).sum(1)
        
        # concat the embedding of support set and query set
        Zs, Zq = proto_centers, Z[n_ways*num_support:]
        # Zs shape: (n_ways, num_hiddens, w, h) => (n_ways * num_query, n-ways, num_hiddens, w, h)
        Zs = Zs.unsqueeze(0).expand(n_ways * num_query, n_ways, -1, -1, -1)
        # Zq shape: (n_ways * num_query, num_hiddens, w, h) => (n_ways * num_query, n_ways, num_hiddens, w, h)
        Zq = Zq.unsqueeze(1).expand(n_ways * num_query, n_ways, -1, -1, -1)
        # cats shape: (n_ways * num_query, n_ways, num_hiddens * 2, w, h)
        cats = torch.cat([Zs, Zq], dim=2)
        size = list(cats.size())        # [n_ways * num_query, n_ways, num_hiddens * 2, w, h]
        cats = cats.view(n_ways * num_query * n_ways, *size[2:])  # cats shape: (n_ways * num_query * n_ways, num_hiddens * 2, w, h)
        
        target_inds = torch.arange(0, n_ways).view(n_ways, 1, 1).expand(n_ways, num_query, 1)
        target_inds = target_inds.contiguous().view(n_ways * num_query, -1)
        target_inds = target_inds.long().to(self.device)
        
        # compute the relation scores between samples of query_set and support set
        scores = self.distance_net(cats).view(n_ways * num_query, n_ways)     # shape: (n_ways * num_query, n_ways)
        
        # compute loss by mse for relation net
        # math: \sum_{i=1}^{m}\sum_{j=1}^{n} (r_{i,j}-\(y_{i}==y_{j}))^{2} 
        # matched pairs have similarity 1 and the mismatched pair have similarity 0
        pairs = torch.zeros_like(scores).scatter_(1, target_inds, 1).float().to(self.device)
        loss_val = F.mse_loss(scores, pairs)

        # compute prediction value
        _, y_hat = scores.max(1)
        acc_val = y_hat.eq(target_inds.squeeze(1)).float().mean()
        
        return loss_val, acc_val
    
    def pred(self, Xs, Xq, n_ways):
        self.eval()
        num_support = Xs.size(0) // n_ways
        num_query = Xq.size(0)
        with torch.no_grad():
            X = torch.cat([Xs, Xq], dim=0)
            X = X.to(self.device)
            Z = self.embedding_net(X)
            size = list(Z.size())
        
            # compute the prototypical center point for each class of support set, sum for relation net
            proto_centers = Z[:n_ways*num_support].view(n_ways, num_support, *size[1:]).sum(1)
            
            # concat the embedding of support set and query set
            Zs, Zq = proto_centers, Z[n_ways*num_support:]
            Zs = Zs.unsqueeze(0).expand(num_query, n_ways, -1, -1, -1)
            Zq = Zq.unsqueeze(1).expand(num_query, n_ways, -1, -1, -1)
            cats = torch.cat([Zs, Zq], dim=2)
            size = list(cats.size())        
            cats = cats.view(num_query * n_ways, *size[2:])  
            # compute the distance scores between samples of query_set and support set
            scores = self.distance_net(cats).view(num_query, n_ways)
            _, y_hat = scores.max(1)
            print(scores.cpu())
        
        return y_hat.cpu()