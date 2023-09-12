import os
import torch 
from torch import nn 
import torch.nn.functional as F
import time

from metric_models.nets import EmbeddingNet, BidirectionLSTMEmbedding
from metric_models.base_model import MetricModelBase


class MatchingModel(MetricModelBase):
    """Define the matching model class"""
    def __init__(self, in_channles, out_channels, input_shape, num_hiddens=64, lr=0.001, gpu_ids=[0],
                 use_fce=True, num_fce_hiddens=32, num_fce_layers=1, distance='cosine', 
                 is_train=True, lr_policy='step', **kwargs) -> None:
        super().__init__(lr, gpu_ids, distance=distance, is_train=is_train, lr_policy=lr_policy, **kwargs)
        
        self.use_fce = use_fce
        
        self.embedding_net = EmbeddingNet(
            in_channles, out_channels, num_hiddens, num_layers=4, input_shape=input_shape
        )
        
        self.model_name = 'MatchingNet'
        self.nets.append('embedding')
        
        
        if use_fce:
           self.fce_net = BidirectionLSTMEmbedding(
               input_size=self.embedding_net.out_size, layer_size=[num_fce_hiddens]*num_fce_layers, 
               devicce=self.device, batch_first=True
           )
           self.fce_net.to(self.device)
           self.nets.append('fce')
        
        
        if is_train:
            self.optimizer = torch.optim.Adam(
                self.embedding_net.parameters(), lr=lr
            )
                
    def to_device(self):
        self.embedding_net.to(self.device)
        if self.use_fce:
            self.fce_net.to(self.device)
        
    def _forward(self, Xs, Xq, n_ways):
        """
        The forward function of Matching Model
        num_support and num_query should be 1 for matching network (N-way One-shot)
        """
        self.embedding_net.train()
        num_support = Xs.size(0) // n_ways
        num_query = Xq.size(0) // n_ways
        assert num_support == 1 and num_query == 1
        
        X = torch.cat([Xs, Xq], dim=0)
        X = X.to(self.device)
        
        Z = self.embedding_net(X)
        
        if self.use_fce:
            Z = self.fce_net(Z)
    
        # compute the distance between samples of query_set and support set
        similarities = self._compute_distance(Z[n_ways*num_support:], Z[:n_ways*num_support])
        # print(n_ways, num_support, num_query, Z.size(), similarities.size())

        # compute prediction probabilities
        target_inds = torch.arange(0, n_ways).view(n_ways, 1, 1).expand(n_ways, num_query, 1).long().to(self.device)
        log_p_y = F.log_softmax(similarities, dim=1).view(n_ways, num_query, -1)
        
        # compute loss and accuracy
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
        
        return loss_val, acc_val
    
    def test_on_batch(self, Xs, Xq, n_ways):
        """
        Test teh Prototypical network on a batch
        Parameters:
            Xq (tuple): the support set for training, containing n_ways * num_support samples
            Xs (tuple): the query set for training, containing n_ways * num_query samples
            n-ways (int): the number of classes of support set.
        Ps: The ground truth of samples is not necessary for this task
        """
        self.embedding_net.eval()
        if self.use_fce:
            self.fce_net.eval()
        with torch.no_grad():
            loss, acc = self._forward(Xs, Xq, n_ways)
        return loss.item(), acc.item()
    
    def pred(self, Xs, Xq, n_ways):
        self.embedding_net.eval()
        num_support = Xs.size(0) // n_ways
        assert num_support == 1
        with torch.no_grad():
            X = torch.cat([Xs, Xq], dim=0)
            X = X.to(self.device)
            Z = self.embedding_net(X)
            if self.use_fce:
                Z = self.fce_net(Z)
            # compute the distance between samples of query_set and support set
            similarities = self._compute_distance(Z[n_ways*num_support:], Z[:n_ways*num_support])
            log_p_y = F.log_softmax(similarities, dim=1).view(Xq.size(0), -1)
            print(log_p_y.cpu())
            _, y_hat = log_p_y.max(1)
        
        return y_hat