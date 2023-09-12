import os
import torch 
from torch import nn 
import torch.nn.functional as F

from metric_models.nets import EmbeddingNet
from metric_models.base_model import MetricModelBase


class ProtoModel(MetricModelBase):
    """Define the Prototypical model"""
    def __init__(self, in_channles, out_channels, input_shape, num_hiddens=64, lr=0.001, gpu_ids=[0],
                 distance='euclidean', is_train=True, lr_policy='step', **kwargs) -> None:
        super().__init__(lr, gpu_ids, distance=distance, is_train=is_train, 
                         lr_policy=lr_policy, **kwargs)
        
        self.embedding_net = EmbeddingNet(in_channles, out_channels, num_hiddens, input_shape, num_layers=4)
        self.nets.append('embedding')
        self.model_name = 'ProtoNet'
        
        if is_train:
            self.optimizer = torch.optim.Adam(
                self.embedding_net.parameters(), lr=lr
            )

    def _forward(self, Xs, Xq, n_ways):
        """The forward function of model"""
        self.embedding_net.train()
        num_support = Xs.size(0) // n_ways
        num_query = Xq.size(0) // n_ways
        
        X = torch.cat([Xs, Xq], dim=0)
        X = X.to(self.device)
        
        Z = self.embedding_net(X)
                
        # compute the prototypical center point for each class of support set
        proto_centers = Z[:n_ways*num_support].view(n_ways, num_support, -1).mean(1)
        
        # compute the distance between samples of query_set and proto_centers
        distance = self._compute_distance(Z[n_ways*num_support:], proto_centers)
        # print(Z.size(), distance.size())
        
        # compute prediction probabilities
        log_p_y = F.log_softmax(-distance, dim=1).view(n_ways, num_query, -1)

        # compute loss and accuracy
        target_inds = torch.arange(0, n_ways).view(n_ways, 1, 1).expand(n_ways, num_query, 1).long().to(self.device)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
        
        return loss_val, acc_val
    
    def pred(self, Xs, Xq, n_ways):
        self.embedding_net.eval()
        num_support = Xs.size(0) // n_ways
        assert num_support == 1
        with torch.no_grad():
            X = torch.cat([Xs, Xq], dim=0)
            X = X.to(self.device)
            Z = self.embedding_net(X)
            
            proto_centers = Z[:n_ways*num_support].view(n_ways, num_support, -1).mean(1)
            distance = self._compute_distance(Z[n_ways*num_support:], proto_centers)
            log_p_y = F.log_softmax(-distance, dim=1).view(Xq.size(0), -1)
            _, y_hat = log_p_y.max(1)
            print(log_p_y.cpu())
        
        return y_hat
