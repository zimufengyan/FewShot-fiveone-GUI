import os
import torch
from torch import nn 
import torch.nn.functional as F

from matching_net.base import get_scheduler, init_weights, init_lstm_hiddens
from matching_net.networks import MatchingEmbeddingNet, AttentionClassifier, BidirectionLSTMEmbedding


class MatchingModel:
    """Define the matching model class"""
    def __init__(self, in_channles, out_channels, num_hiddens, lr, gpu_ids: list,
                 input_shape, use_fce=True, num_fce_hiddens=32,
                 distance='euclidean', is_train=True, lr_policy='step', **kwargs) -> None:
        self.distance = distance
        self.is_train = is_train
        self.gpu_ids = gpu_ids
        self.kwargs = kwargs
        self.use_fce = use_fce
        
        self.embedding = MatchingEmbeddingNet(
            in_channles, out_channels, num_hiddens, num_layers=4, input_shape=input_shape
        )
        
        # self.classifier = AttentionClassifier()
        
        self.device = torch.device(
            'cuda:{}'.format(self.gpu_ids[0])
            ) if self.gpu_ids and torch.cuda.is_available() else torch.device('cpu')  # get device name: CPU or GPU
        self.embedding.to(self.device)
        
        if use_fce:
           self.fce = BidirectionLSTMEmbedding(
               input_size=self.embedding.out_size, layer_size=[32], 
               devicce=self.device, batch_first=True
           )
           self.fce.to(self.device)
        
        
        if is_train:
            self.optimizer = torch.optim.Adam(
                self.embedding.parameters(), lr=lr
            )
            if lr_policy is not None or \
            (isinstance(lr_policy, str) and lr_policy.lower() != 'none'):
                self.scheduler = get_scheduler(self.optimizer, lr_policy, **kwargs)
            else:
                self.scheduler = None
                
    def to_device(self):
        self.embedding.to(self.device)
        if self.use_fce:
            self.fce.to(self.device)
        
    def init_net(self, init_type='normal', init_gian=0.2):
        """Initialize the weights of embedding net"""
        init_weights(self.embedding, init_type=init_type, init_gain=init_gian)
        
    def _compute_distance(self, x, y):
        """Conpute the distance between x and y"""
        if x.size(1) != y.size(1):
            raise RuntimeError(f"The size of tensor x ({x.size(1)}) must match the size of tensor y ({y.size(1)}) at dimension 1")
        n, m = x.size(0), y.size(0)
        d = x.size(1)   
        
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        if self.distance == 'euclidean':
            return torch.pow((x - y), 2).sum(2)
        elif self.distance == 'cosine':
            return F.cosine_similarity(x, y, dim=2)
        else:
            raise ValueError(f"unsported distance function ({self.distance})")
        
    def _forward(self, Xs, Xq, n_ways):
        """
        The forward function of Matching Model
        num_support and num_query should be 1 for matching network (N-way One-shot)
        """
        self.embedding.train()
        num_support = Xs.size(0) // n_ways
        num_query = Xq.size(0) // n_ways
        assert num_support == 1 and num_query == 1
        
        X = torch.cat([Xs, Xq], dim=0)
        X = X.to(self.device)
        
        Z = self.embedding(X)
        
        if self.use_fce:
            Z = self.fce(Z)
    
        # compute the distance between samples of query_set and proto_centers
        similarities = self._compute_distance(Z[n_ways*num_support:], Z[:n_ways*num_support])
        # print(n_ways, num_support, num_query, Z.size(), similarities.size())

        # compute prediction probabilities
        target_inds = torch.arange(0, n_ways).view(n_ways, 1, 1).expand(n_ways, num_query, 1).long().to(self.device)
        log_p_y = F.log_softmax(similarities, dim=1).view(n_ways, num_query, -1)
        
        # compute loss and accuracy
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
        # target_inds = torch.arange(0, n_ways).view(n_ways, 1).expand(n_ways, num_support).view(n_ways*num_support).long().to(self.device)
        # target_one_hot = F.one_hot(target_inds).float()
        # preds = self.classifier(similarities, target_one_hot)
        # values, indices = preds.max(1)
        # acc_val = torch.mean((indices.squeeze() == target_inds).float())
        # loss_val = F.cross_entropy(preds, target_inds.long())
        
        return loss_val, acc_val
    
    def train_on_batch(self, Xs, Xq, n_ways):
        """
        Train the Prototypical network on a batch.
        Parameters:
            Xq (tuple): the support set for training, containing n_ways * num_support samples
            Xs (tuple): the query set for training, containing n_ways * num_query samples
            n-ways (int): the number of classes of support set.
        Ps: The ground truth of samples is not necessary for this task
        """
        if not self.is_train: return
        loss, acc = self._forward(Xs, Xq, n_ways)
        
        # backward net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc.item()
    
    def test_on_batch(self, Xs, Xq, n_ways):
        """
        Test teh Prototypical network on a batch
        Parameters:
            Xq (tuple): the support set for training, containing n_ways * num_support samples
            Xs (tuple): the query set for training, containing n_ways * num_query samples
            n-ways (int): the number of classes of support set.
        Ps: The ground truth of samples is not necessary for this task
        """
        self.embedding.eval()
        with torch.no_grad():
            loss, acc = self._forward(Xs, Xq, n_ways)
        return loss.item(), acc.item()
        
    def update_lr(self, verbose=1):
        """Update learning rate"""
        old_lr = self.optimizer.param_groups[0]['lr']
        if self.scheduler is not None:
            self.scheduler.step()
        new_lr = self.optimizer.param_groups[0]['lr']  
        if new_lr != old_lr and verbose:
            print(f"Updating learning rate: {old_lr} -> {new_lr}")
            
    def save_networks(self, save_dir, name):
        torch.save(self.embedding.state_dict(), os.path.join(save_dir, name + '.pth'))