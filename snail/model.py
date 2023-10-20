import os
import numpy as np
import torch
from torch import nn 

from snail.utils import *
from snail.networks import *


class SNAIL(nn.Module):
    """Define Snail model"""
    def __init__(self, num_embedding, embedding_net, n_way, k_shot, device, bias=True):
        super().__init__()
        self.embedding_net = embedding_net
        self.n_way = n_way
        self.k_shot = k_shot
        self.seq_length = n_way * k_shot + 1
        
        self.device = device
        
        in_channels = num_embedding + n_way
        self.attention_1 = AttentionBlock(in_channels, 64, 32)
        self.tc_1 = TCBlock(self.attention_1.out_channels, self.seq_length, 128, bias=bias)
        self.attention_2 = AttentionBlock(self.tc_1.out_channels, 256, 128)
        self.tc_2 = TCBlock(self.attention_2.out_channels, self.seq_length, 128, bias=bias)
        self.attention_3 = AttentionBlock(self.tc_2.out_channels, 512, 256)
        
        # self.conv = nn.Conv2d(self.attention_3.out_channels, n_way, 1, 1, 0, bias=bias)
        self.fc = nn.Linear(self.attention_3.out_channels, n_way)
    
    def forward(self, X, y):
        X = self.embedding_net(X)   # size: (batch_size, C*H*W)
        b_size = X.size(0) // self.seq_length
        last_idxs = [(i + 1) * (self.n_way * self.k_shot + 1) - 1 for i in range(b_size)]
        y[last_idxs] = torch.tensor(np.zeros((b_size, y.size(1))), device=self.device).type(y.dtype)
        
        Z = torch.cat([X, y], dim=1)    # size: (batch_size, C*H*W + n_way)
        Z = Z.view(b_size, self.seq_length, -1)     # size: (b_size, T, D), let `T` equals to `seq_length`
        Z = self.attention_1(Z).permute(0, 2, 1)         # size: (b_size, D + 32, T)
        Z = self.tc_1(Z).permute(0, 2, 1)    # size: (b_size, T, D + 32 + [log(T, 2)]*128)
        Z = self.attention_2(Z).permute(0, 2, 1) # size (b_size, D + 32 + [log(T, 2)]*128 + 128, T)
        Z = self.tc_2(Z).permute(0, 2, 1)    # size: (b_size, T, D + 32 + 2*[log(T, 2)]*128 + 128)
        Z = self.attention_3(Z)# size: (b_size, T, D + 32 + 2*[log(T, 2)]*128 + 128 + 256)
        
        logits = self.fc(Z)   # size: (b_size, T, n_way)
        
        return logits
        
        
class SnailModel:
    def __init__(self, in_channels=1, n_way=5, k_shot=1, dataset_name='Omniglot',
                 lr=0.001, gpu_ids=[0], is_train=True, dropout=0.9, bias=True, **kwargs):
        self.is_train = is_train
        self.gpu_ids = gpu_ids
        self.kwargs = kwargs
        self.dataset_name = dataset_name
        self.n_way = n_way
        self.k_shot = k_shot
        self.seq_length = n_way * k_shot + 1
        
        self.model_name = 'SNAIL'
        self.nets = ['snail']
        
        self.device = torch.device(
            'cuda:{}'.format(self.gpu_ids[0])
        ) if self.gpu_ids and torch.cuda.is_available() else torch.device('cpu')  # get device name: CPU or GPU
           
        if dataset_name.lower() == "omniglot":
            num_embedding = 64
            embedding_net = OmniglotEmbedding(
                in_channels=1, out_channels=num_embedding, num_hiddens=64, bias=bias
            )
        elif dataset_name.lower() == "mini_imagenet":
            num_embedding = 384
            embedding_net = MiniImageEmbedding(in_channels=3, dropout=dropout, bias=bias)
        else:
            num_embedding = 384
            embedding_net = MiniImageEmbedding(in_channels=in_channels, dropout=dropout, bias=bias)
        
        self.snail_net = SNAIL(num_embedding, embedding_net, n_way, k_shot, self.device, bias)
        
        self.criterion = nn.CrossEntropyLoss()
        if self.is_train:
            self.optimizer = torch.optim.Adam(
                self.snail_net.parameters(), lr=lr, betas=[0.5, 0.99]
            )
            
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
            init_weights(net, init_type=init_type, init_gain=init_gian)
            
    def train_on_batch(self, X, y):
        """
        Train the Snail network on a batch.
        Parameters:
            X (Tensor): the training samples set, size of (B, C, H, W), where B = batch_size * (N*K+1)
            y (Tensor): the onehot of ground truth of `X`, size of (B, N)
        """
        if not self.is_train: return
        self.train()

        loss, acc, _ = self._forward(X, y)
        
        # backward net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), acc.item()
    
    def test_on_batch(self, X, y):
        """
        Test teh Snail network on a batch
        Parameters:
            X (Tensor): the training samples set, size of (B, C, H, W), where B = batch_size * (N*K+1)
            y (Tensor): the onehot of ground truth of `X`, size of (B, N)
        """
        self.eval()
        
        with torch.no_grad():
            loss, acc, _ = self._forward(X, y)
        return loss.item(), acc.item() 
    
    def pred(self, X, y):
        """
        Pred y at timestep i * (N*K + 1), i=0, 1, batch_size - 1
        Parameters:
            X (Tensor): the training samples set, size of (B, C, H, W), where B = batch_size * (N*K+1)
            y (Tensor): the onehot of ground truth of `X`, size of (B, N)
        """
        self.eval()
        with torch.no_grad():
            _, _, y_hat = self._forward(X, y)
        return y_hat.cpu()
    
    def _forward(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        b_size = y.size(0) // self.seq_length
        last_idxs = [(i + 1) * self.seq_length - 1 for i in range(b_size)]
        last_targets = y[last_idxs].argmax(dim=-1).to(self.device)  # labels at timestamp N*K+1
        logits = self.snail_net(X, y)
        last_logits = logits[:, -1, :]  # size (b_size, n_way)  # compute loss at timestamp N*K+1
        print(last_logits.cpu())
        
        loss = self.criterion(last_logits, last_targets)
        
        pred = last_logits.argmax(dim=-1)
        acc = pred.eq(last_targets).float().mean()
        
        return loss, acc, pred
        
    def update_lr(self, verbose=1):
        """Update learning rate"""
        old_lr = self.optimizer.param_groups[0]['lr']
        if self.scheduler is not None:
            self.scheduler.step()
        new_lr = self.optimizer.param_groups[0]['lr']  
        if new_lr != old_lr and verbose:
            print(f"Updating learning rate: {old_lr} -> {new_lr}")
            
    def save_networks(self, save_dir, epoch):
        """
        Save all the networks to the disk.
        Parameters:
            save_dir (str) -- directory used to save networks
            epoch (int) -- current epoch; used in the file name '{}_net_on_{}_{}.pth'.format (name, task_name, epoch)
        """
        target_dir = os.path.join(save_dir, self.model_name)
        if not os.path.exists(target_dir): 
            os.mkdir(target_dir)
        for name in self.nets:
            net = getattr(self, name + '_net')
            save_name = f'{name}_net_on_{self.dataset_name}_{epoch}.pth'
            torch.save(net.cpu().state_dict(), os.path.join(target_dir, save_name))

    def load_networks(self, load_dir, epoch):
        """
        Load all the networks from the disk.
        Parameters:
            load_dir (str) -- directory that stores networks
            epoch (int) -- current epoch; used in the file name '{}_net_on_{}_{}.pth'.format (name, dataset_name, epoch)
        """
        target_dir = os.path.join(load_dir, self.model_name)
        for name in self.nets:
            net = getattr(self, name + '_net')
            load_name = f'{name}_net_on_{self.dataset_name}_{epoch}.pth'
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