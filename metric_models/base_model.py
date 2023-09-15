import os
import torch 
from torch import nn 
import torch.nn.functional as F

from metric_models.base import get_scheduler, init_weights

class MetricModelBase:
    """Define the metric learning basemodel"""
    def __init__(self, gpu_ids=[0], distance='euclidean', is_train=True, **kwargs) -> None:
        self.distance = distance
        self.is_train = is_train
        self.gpu_ids = gpu_ids
        self.kwargs = kwargs
        
        self.embedding_net = None       # all subclass should have this attribution
        self.device = torch.device(
            'cuda:{}'.format(self.gpu_ids[0])
            ) if self.gpu_ids and torch.cuda.is_available() else torch.device('cpu')  # get device name: CPU or GPU
        
        self.optimizer = None
        self.scheduler = None
        
        self.model_name = "" 
        self.nets = []      # store network name
            
    def to_device(self):
        for name in self.nets:
            net = getattr(self, name + '_net')
            net.to(self.device)
        
    def load_scheduler(self, lr_policy='step', **kwargs):
        """Load scheduler"""
        if self.optimizer is None: return
        if (isinstance(lr_policy, str) and lr_policy.lower() != 'none'):
            self.scheduler = get_scheduler(self.optimizer, lr_policy, **kwargs)            
        
    def init_net(self, init_type='normal', init_gian=0.2):
        """Initialize all nets"""
        for name in self.nets:
            net = getattr(self, name + '_net')
            init_weights(net, init_type=init_type, init_gain=init_gian)
            
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
        """The forward function of model"""
        raise NotImplementedError
    
        
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
        self.train()
        loss, acc = self._forward(Xs, Xq, n_ways)
        
        # backward net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc.item()
    
    def test_on_batch(self, Xs, Xq, n_ways):
        """
        Test the Prototypical network on a batch
        Parameters:
            Xq (Tensor): the support set for test, containing n_ways * num_support samples
            Xs (Tensor): the query set for test, containing n_ways * num_query samples
            n-ways (int): the number of classes of support set.
        Ps: The ground truth of samples is not necessary for this task
        """
        self.eval()
        with torch.no_grad():
            loss, acc = self._forward(Xs, Xq, n_ways)
        return loss.item(), acc.item()
    
    def pred(self, Xs, Xq, n_ways):
        """
        Perfrom a prediction on a N-way-K-shot task

        Args:
            Xs (Tensor): Support set, size of (N-way * num_support, c, w, d)
            Xq (Tensor): Query set, size of (num_query, c, w, d)
            n_ways (int): N-way

        Returns:
            Tensor: Pred value of each query sample
        """
        raise NotImplementedError
        
    def update_lr(self, verbose=1):
        """Update learning rate"""
        old_lr = self.optimizer.param_groups[0]['lr']
        if self.scheduler is not None:
            self.scheduler.step()
        new_lr = self.optimizer.param_groups[0]['lr']  
        if new_lr != old_lr and verbose:
            print(f"Updating learning rate: {old_lr} -> {new_lr}")
    
    def save_networks(self, save_dir, task_name, epoch):
        """
        Save all the networks to the disk.
        Parameters:
            save_dir (str) -- directory used to save networks
            task_name (str) -- current dataset name; used in the file name '{}_net_on_{}_{}.pth'.format (name, task_name, epoch)
            epoch (int) -- current epoch; used in the file name '{}_net_on_{}_{}.pth'.format (name, task_name, epoch)
        """

        target_dir = os.path.join(save_dir, self.model_name)
        if not os.path.exists(target_dir): 
            os.mkdir(target_dir)
        for name in self.nets:
            net = getattr(self, name + '_net')
            save_name = f'{name}_net_on_{task_name}_{epoch}.pth'
            torch.save(net.cpu().state_dict(), os.path.join(target_dir, save_name))

    def load_networks(self, load_dir, task_name, epoch):
        """
        Load all the networks from the disk.
        Parameters:
            load_dir (str) -- directory that stores networks
            task_name (str) -- current dataset name; used in the file name '{}_net_on_{}_{}.pth'.format (name, task_name, epoch)
            epoch (int) -- current epoch; used in the file name '{}_net_on_{}_{}.pth'.format (name, task_name, epoch)
        """
        target_dir = os.path.join(load_dir, self.model_name)
        for name in self.nets:
            net = getattr(self, name + '_net')
            load_name = f'{name}_net_on_{task_name}_{epoch}.pth'
            load_path = os.path.join(target_dir, load_name)
            state = torch.load(load_path, map_location=str(self.device))
            if hasattr(state, '_metadata'):
                del state._metadata
            net.load_state_dict(state)
            
    def train(self):
        """Switch model to train mode"""
        for name in self.nets:
            net = getattr(self, name + '_net')
            net.train()
        
    def eval(self):
        """Switch model to eval mode"""
        for name in self.nets:
            net = getattr(self, name + '_net')
            net.eval()
        