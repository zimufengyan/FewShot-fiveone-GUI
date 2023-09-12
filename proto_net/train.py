import torch 
from torch.utils.data import DataLoader
import os
import time
import numpy as np
import math
from torchvision import datasets
from torchvision.transforms import transforms, InterpolationMode
import matplotlib.pyplot as plt

from model import ProtoModel
from options import get_parser
from sampler import FewShotBatchSampler


def init_seed(opt):
    if opt.manual_seed == -1: return
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    

def init_dataloader(opt):
    if opt.dataset_name == 'Omniglot':
        train_transform = transforms.Compose([
        transforms.Resize((28, 28), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomApply(
                [transforms.RandomAffine(degrees=15, shear=0.3 * 180 / math.pi, scale=(0.8, 2.0))],
                p=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.87, std=0.33)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((28, 28), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.87, std=0.33)
        ])
        train_dataset = datasets.Omniglot(root=opt.dataset_root, background=True, transform=train_transform, download=bool(opt.download))
        test_dataset = datasets.Omniglot(root=opt.dataset_root, background=False, transform=test_transform, download=bool(opt.download))
    elif opt.dataset_name == 'mini_Imagenet':
        raise NotImplementedError
    else:
        raise ValueError(f"The parameter 'dataset_name' should be 'Omniglot' or 'mini_Imagenet', rather than {opt.dataset_name}")
    train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    test_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
    
    n_ways_tr, n_ways_ts = opt.n_classes_per_iter_training, opt.n_classes_per_iter_test
    sTr, qTr = opt.support_samples_per_class_training, opt.query_samples_per_class_training
    sTs, qTs = opt.support_samples_per_class_test, opt.query_samples_per_class_test
    train_sampler = FewShotBatchSampler(train_labels, n_ways_tr, sTr + qTr, opt.iterations)
    test_sampler = FewShotBatchSampler(test_labels, n_ways_ts, sTs + qTs, opt.iterations)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
    
    return train_loader, test_loader


def split_support_query(X, n_ways, n_support, n_query):
    assert X.size(0) == n_ways * (n_support + n_query)
    s_idxs = torch.LongTensor(np.arange(0, X.size(0), n_support + n_query))
    q_idxs = torch.LongTensor(np.arange(n_support, X.size(0), n_support + n_query))
    Xs = X[s_idxs]
    Xq = X[q_idxs]
    
    return Xs, Xq


if __name__ == '__main__':
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    best_evaluation_accuracy = 0.0
    best_evaluation_epoch = 0
    
    # initialize argments
    opt = get_parser().parse_args()
    init_seed(opt)
    
    # initialize folders 
    if not os.path.exists(opt.dataset_root):
        os.mkdir(opt.dataset_root)
    if not os.path.exists(opt.plot_dir):
        os.mkdir(opt.plot_dir)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    
    # initialize dataloader
    print('[INFO] Loading data')
    train_loader, test_loader = init_dataloader(opt)
    
    # initialize model
    print('[INFO] Building model')
    if opt.dataset_name == 'Omniglot':
        in_channels = 1
    elif opt.dataset_name == 'mini_Imagenet':
        in_channels = 3
    else:
        raise ValueError(f"The parameter 'dataset_name' should be 'Omniglot' or 'mini_Imagenet', rather than {opt.dataset_name}")
    gpu_ids = opt.gpu_ids.split(',')
    model = ProtoModel(
        in_channels, opt.out_channels, opt.n_hiddens, opt.learning_rate, gpu_ids, is_train=True, 
        lr_policy=opt.lr_scheduler_policy, gamma=opt.lr_gamma, lr_decay_iters=opt.lr_decay_epochs, distance=opt.distance_fn
    )
    model.init_net(opt.init_type, opt.init_gain)
    model.to_device()
    
    n_ways_tr, n_ways_ts = opt.n_classes_per_iter_training, opt.n_classes_per_iter_test
    sTr, qTr = opt.support_samples_per_class_training, opt.query_samples_per_class_training
    sTs, qTs = opt.support_samples_per_class_test, opt.query_samples_per_class_test
    
    print('[INFO] Starting training')
    for epoch in range(opt.n_epochs):
        st = time.time()
        train_loss_mean, train_acc_mean, train_size = 0.0, 0.0, 0
        for X, _ in train_loader:
            Xs, Xq = split_support_query(X, n_ways_tr, sTr, qTr)
            train_loss, train_acc = model.train_on_batch(Xs, Xq, n_ways_tr)
            b_size = Xq.size(0)
            train_size += b_size
            train_loss_mean += train_loss * b_size
            train_acc_mean += train_acc * b_size
        train_losses.append(train_loss_mean / train_size)
        train_accuracies.append(train_acc_mean / train_size)
        model.update_lr()
        
        test_loss_mean, test_acc_mean, test_size = 0.0, 0.0, 0
        for X, _ in test_loader:
            Xs, Xq = split_support_query(X, n_ways_ts, sTs, qTs)
            test_loss, test_acc = model.test_on_batch(Xs, Xq, n_ways_ts)  
            b_size = Xq.size(0)     
            test_size += b_size
            test_loss_mean += test_loss * b_size
            test_acc_mean += test_acc * b_size
        test_losses.append(test_loss_mean / test_size)
        test_accuracies.append(test_acc_mean / test_size) 
        
        if epoch % opt.print_freq == 0 or epoch == opt.n_epochs - 1:
            print(f"Epoch {epoch+1}/{opt.n_epochs} {time.time() - st :.2f}sec :", end=' ')
            print(f"Train Loss: {train_losses[epoch] :.4f}  Train Accuracy: {train_accuracies[epoch] :.4f}", end="\t")
            print(f"Test Loss: {test_losses[epoch] :.4f}  Test Accuracy: {test_accuracies[epoch] :.4f}")
        if test_accuracies[epoch] > best_evaluation_accuracy:
            best_evaluation_accuracy = test_accuracies[epoch]
            best_evaluation_epoch = epoch
            
    print("[INFO] End training")
    print(f'The best evaluation accuracy is {best_evaluation_accuracy :.4f} at epoch {best_evaluation_epoch}')
    print(f'[INFO] The model has saved to {opt.save_dir}')
    
    model.save_networks(save_dir=opt.save_dir, epoch=opt.n_epochs + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes.flatten()
    t = list(range(1, opt.n_epochs + 1))
    ax1.plot(t, train_losses, label='Train Loss')
    ax1.plot(t, test_losses, label='Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_xlim(1, opt.n_epochs)
    ax1.legend()
    ax2.plot(t, train_accuracies, label='Train Accuracy')
    ax2.plot(t, test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(1, opt.n_epochs)
    ax2.legend()
    plt.savefig(os.path.join(opt.plot_dir, f'protonet_metrics_on_{opt.dataset_name}.png'), dpi=150)
    