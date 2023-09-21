import os
import torch
from torch import nn 
import torch.nn.functional as F
from collections import OrderedDict

from meta_models.base import get_scheduler, init_weights, compute_accuracy
from meta_models.base_model import MetaModelBase


class ModelAgnosticMetaLearning(MetaModelBase):
    """Define MAML class"""
    def __init__(self, learner: nn.Module, meta_lr=0.001, train_lr=0.2, test_lr=0.2, train_inner_step=1, test_inner_step=3, 
                 num_classes=None, gpu_ids=[0], is_classify_task=True, is_train=True) -> None:
        super().__init__(meta_lr, train_lr, test_lr, train_inner_step, test_inner_step, num_classes, gpu_ids, 
                         is_classify_task, is_train)
        
        self.learner_net = learner
        self.model_name = 'MAML'
        self.nets = ['learner']
        
        self.device = torch.device(
            'cuda:{}'.format(self.gpu_ids[0])
            ) if self.gpu_ids and torch.cuda.is_available() else torch.device('cpu')  # get device name: CPU or GPU
        
        if is_train:
            self.optimizer = torch.optim.Adam(
                self.learner_net.parameters(), lr=meta_lr
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
        
    def adapt(self, Xs, ys, step, lr):
        fast_weights = OrderedDict(self.learner_net.named_parameters())
        for step in range(step):
            logits = self.learner_net.functional_forward(Xs, params=fast_weights)
            support_loss = self.criterion(logits, ys)
            grads = torch.autograd.grad(
                support_loss, fast_weights.values(), create_graph=False
            )
            fast_weights = OrderedDict((name, param - lr * grad)
                                                   for ((name, param), grad) in zip(fast_weights.items(), grads))

        return fast_weights
    
    def pred(self, Xs, ys, Xq, step=None, lr=None):
        # fine-tuning on Xs and ys 
        if step is None:
            step = self.test_inner_step
        if lr is None:
            lr = self.test_lr
        self.eval()
        Xs, ys, Xq = Xs.to(self.device), ys.to(self.device), Xq.to(self.device)
        if len(Xq.size()) < 4:
            Xq = Xq.unsqueeze(0)    # add batch axis
        params = self.adapt(Xs, ys, step, lr)
        with torch.no_grad():
            logits = self.learner_net.functional_forward(Xq, params)
            logits = F.softmax(logits, dim=1)
        print(logits.cpu())
        if self.is_classify_task:
            _, y_hat = logits.max(1)
            return y_hat.cpu()
        return logits.cpu()
            
FOMAML = ModelAgnosticMetaLearning