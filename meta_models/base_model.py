import os
import torch 
from torch import nn 
import torch.nn.functional as F

from metric_models.base import get_scheduler, init_weights

class MetaModelBase:
    """Define the meta learning basemodel"""
    def __init__(self, meta_lr=0.001, train_lr=0.4, test_lr=0.4, train_inner_step=1, test_inner_step=3, 
                 num_classes=None, gpu_ids=[0], is_classify_task=True, is_train=True) -> None:
        self.num_classes = num_classes
        self.is_classify_task = is_classify_task
        self.is_train = is_train
        self.train_lr = train_lr
        self.test_lr = test_lr
        self.train_inner_step = train_inner_step
        self.test_inner_step = test_inner_step  
        self.gpu_ids = gpu_ids
        
        self.device = torch.device(
            'cuda:{}'.format(self.gpu_ids[0])
            ) if self.gpu_ids and torch.cuda.is_available() else torch.device('cpu')  # get device name: CPU or GPU
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.model_name = "" 
        self.nets = []      # store network name
            
    def load_scheduler(self, lr_policy='step', **kwargs):
        """Load scheduler"""
        if self.optimizer is None: return
        if (isinstance(lr_policy, str) and lr_policy.lower() != 'none'):
            self.scheduler = get_scheduler(self.optimizer, lr_policy, **kwargs)       
            
    def to_device(self):
        for name in self.nets:
            net = getattr(self, name + '_net')
            net.to(self.device)
        
    def init_net(self, init_type='normal', init_gian=0.2):
        """Initialize the all nets"""
        for name in self.nets:
            net = getattr(self, name + '_net')
            net.to(self.device)
            init_weights(net, init_type=init_type, init_gain=init_gian)
    
    def train_on_batch(self, support_sets, query_sets):
        """
        Train model on a batch data. A batch consists of multi tasks which contain support set and query set.
        Parameters:
            support_sets (List[Tensor]): a batch of support set for each task, containing samples and labels
            query_sets (List[Tensor]): a batch of query set for each task, containing samples and labels
        """
        if not self.is_train: return
        self.train()
        result = self._forward(support_sets, query_sets, self.train_inner_step, self.train_lr)
        
        # update slow-weight
        outer_loss = result['mean_outer_loss']
        self.optimizer.zero_grad()
        outer_loss.backward()
        self.optimizer.step()
        
        if self.is_classify_task:
            return result['mean_outer_loss'].cpu().item(), result['mean_accuracy'].cpu().item()
        else:
            return result['mean_outer_loss'].cpu().item()
        
    def test_on_batch(self, support_sets, query_sets):
        """
        Test model on a batch data. A batch consists of multi tasks which contain support set and query set.
        Parameters:
            support_sets (Tensor): a batch of support set for each task, containing samples and labels
            query_sets (Tensor): a batch of query set for each task, containing samples and labels
        """
        self.eval()
        result = self._forward(support_sets, query_sets, self.test_inner_step, self.test_lr)
        if self.is_classify_task:
            return result['mean_outer_loss'].cpu().item(), result['mean_accuracy'].cpu().item()
        else:
            return result['mean_outer_loss'].cpu().item()
        
    def _forward(self, support_sets, query_sets, inner_step, lr):
        raise NotImplementedError
        
    def pred(self, Xs, ys, Xq, step=None, lr=None):
        """
        Perfrom a prediction on a N-way-K-shot task

        Args:
            Xs (Tensor): support set, size of (N-way * num_support, c, w, d)
            ys (Tensor): labels of support set
            Xq (Tensor): query set, size of (num_query, c, w, d)
            step (int): perform 'step' inner step, i.e., gradient descent. when 'step' is None, use self.test_inner_step instead
            lr (float): learning rate for inner step. when 'lr' is None, use self.test_lr instead

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
        