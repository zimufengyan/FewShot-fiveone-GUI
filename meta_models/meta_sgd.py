import os
import torch
from torch import nn 
import torch.nn.functional as F
from collections import OrderedDict
from itertools import chain

import sys

from meta_models.utils import *
from meta_models.base_model import MetaModelBase


class MetaSGD(MetaModelBase):
    """Define MAML class"""
    def __init__(self, learner: nn.Module, meta_lr=0.001, train_inner_step=1, test_inner_step=3, 
                 num_classes=None, gpu_ids=[0], is_classify_task=True, is_train=True) -> None:
        self.num_classes = num_classes
        self.is_classify_task = is_classify_task
        self.is_train = is_train
        self.train_inner_step = train_inner_step
        self.test_inner_step = test_inner_step  
        self.gpu_ids = gpu_ids
        
        self.task_lr = OrderedDict()    # lr matrix that needs to be learned
        
        self.learner_net = learner
        self.model_name = 'MetaSGD'
        self.nets = ['learner']
        
        self.device = torch.device(
            'cuda:{}'.format(self.gpu_ids[0])
            ) if self.gpu_ids and torch.cuda.is_available() else torch.device('cpu')  # get device name: CPU or GPU
        
        if is_train:
            # different from MAML
            self.init_task_lr()
            self.optimizer = torch.optim.Adam(
                chain(self.learner_net.parameters(), self.task_lr.values()), lr=meta_lr
            )
        if self.is_classify_task:
            # for classify task
            if self.num_classes is not None and self.num_classes > 2:
                self.criterion = nn.CrossEntropyLoss()
            elif self.num_classes is not None:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                raise ValueError("The param 'num_classes' must be specified when param 'is_classify_task' eqals 'True'")
        else:
            # for regression task
            self.criterion = nn.MSELoss()
                
    def init_task_lr(self, init_task_lr=1e-3):
        """Initialize lr matrix"""
        for k, v in self.learner_net.named_parameters():
            self.task_lr[k] = nn.Parameter(
                init_task_lr * torch.ones_like(v, requires_grad=True, device=self.device)
            )
            
    def train_on_batch(self, support_sets, query_sets):
        """
        Train model on a batch data. A batch consists of multi tasks which contain support set and query set.
        Parameters:
            support_sets (List[Tensor]): a batch of support set for each task, containing samples and labels
            query_sets (List[Tensor]): a batch of query set for each task, containing samples and labels
        """
        if not self.is_train: return
        self.train()
        result = self._forward(support_sets, query_sets, self.train_inner_step, self.task_lr)
        
        # update slow-weight
        meta_loss = result['mean_meta_loss']
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        
        if self.is_classify_task:
            return result['mean_meta_loss'].cpu().item(), result['mean_accuracy'].cpu().item()
        else:
            return result['mean_meta_loss'].cpu().item()
        
    def test_on_batch(self, support_sets, query_sets):
        """
        Test model on a batch data. A batch consists of multi tasks which contain support set and query set.
        Parameters:
            support_sets (Tensor): a batch of support set for each task, containing samples and labels
            query_sets (Tensor): a batch of query set for each task, containing samples and labels
        """
        self.eval()
        result = self._forward(support_sets, query_sets, self.test_inner_step, self.task_lr)
        if self.is_classify_task:
            return result['mean_meta_loss'].cpu().item(), result['mean_accuracy'].cpu().item()
        else:
            return result['mean_meta_loss'].cpu().item()
        
    def _forward(self, support_sets, query_sets, inner_step, lr):
        num_tasks = support_sets[-1].size(0)
        if query_sets[-1].size(0) != num_tasks:
            raise RuntimeError(f"The size ({query_sets.size(0)}) of query sets must match the size ({num_tasks}) of support sets at dim 0")
        
        result = {
            'mean_outer_loss': torch.tensor(0., device=self.device)
        }
        if self.is_classify_task:
            result['mean_accuracy'] = torch.tensor(0., device=self.device)
        for Xs, ys, Xq, yq in zip(*support_sets, *query_sets):
            Xs, Xq = Xs.to(self.device), Xq.to(self.device)
            ys, yq = ys.to(self.device), yq.to(self.device)
            
            # update fast-weight
            fast_weights = self.adapt(Xs, ys, inner_step,  lr)
            
            with torch.set_grad_enabled(self.learner_net.training):
                query_out = self.learner_net.functional_forward(Xq, fast_weights)
                query_loss = self.criterion(query_out, yq)
                result['mean_outer_loss'] += query_loss
                
            if self.is_classify_task:
                acc = compute_accuracy(query_out, yq)
                result['mean_accuracy'] += acc
                
        result['mean_outer_loss'] = result['mean_outer_loss'].div_(num_tasks)   
        if self.is_classify_task:
            result['mean_accuracy'] = result['mean_accuracy'].div_(num_tasks)      
        return result
        
    def adapt(self, Xs, ys, step, lr: OrderedDict):
        fast_weights = OrderedDict(self.learner_net.named_parameters())
        for step in range(step):
            logits = self.learner_net.functional_forward(Xs, params=fast_weights)
            support_loss = self.criterion(logits, ys)
            grads = torch.autograd.grad(
                support_loss, fast_weights.values(), create_graph=False
            )
            # different from MAML
            fast_weights = OrderedDict((name, param - lr[name] * grad)
                                                   for ((name, param), grad) in zip(fast_weights.items(), grads))

        return fast_weights
    
    def pred(self, Xs, ys, Xq, step=None):
        # fine-tuning on Xs and ys 
        if step is None:
            step = self.test_inner_step
        self.eval()
        Xs, ys, Xq = Xs.to(self.device), ys.to(self.device), Xq.to(self.device)
        if len(Xq.size()) < 4:
            Xq = Xq.unsqueeze(0)    # add batch axis
        params = self.adapt(Xs, ys, step, self.task_lr)
        with torch.no_grad():
            logits = self.learner_net.functional_forward(Xq, params)
            logits = F.softmax(logits, dim=1)
        print(logits.cpu())
        if self.is_classify_task:
            _, y_hat = logits.max(1)
            return y_hat.cpu()
        return logits.cpu()
    
    def save_networks(self, save_dir, task_name, epoch, save_task_lr=True):
        """
        Save all the networks to the disk.
        Parameters:
            save_dir (str) -- directory used to save networks
            task_name (str) -- current dataset name; used in the file name '{}_net_on_{}_{}.pth'.format (name, task_name, epoch)
            epoch (int) -- current epoch; used in the file name '{}_net_on_{}_{}.pth'.format (name, task_name, epoch)
            save_task_lr (bool) -- Whether or not need to save task lr
        """
        target_dir = os.path.join(save_dir, self.model_name)
        if not os.path.exists(target_dir): 
            os.mkdir(target_dir)
        for name in self.nets:
            net = getattr(self, name + '_net')
            save_name = f'{name}_net_on_{task_name}_{epoch}.pth'
            torch.save(net.cpu().state_dict(), os.path.join(target_dir, save_name))
        if save_task_lr:
            save_name = f'task_lr_on_{task_name}_{epoch}.pth'
            torch.save(self.task_lr, os.path.join(target_dir, save_name))

    def load_networks(self, load_dir, task_name, epoch, load_task_lr=True):
        """
        Load all the networks from the disk.
        Parameters:
            load_dir (str) -- directory that stores networks
            task_name (str) -- current dataset name; used in the file name '{}_net_on_{}_{}.pth'.format (name, task_name, epoch)
            epoch (int) -- current epoch; used in the file name '{}_net_on_{}_{}.pth'.format (name, task_name, epoch)
            load_task_lr (bool) -- Whether or not need to load task lr
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
        if load_task_lr:
            load_name = f'task_lr_on_{task_name}_{epoch}.pth'
            self.task_lr = torch.load(load_path, map_location=str(self.device))
            