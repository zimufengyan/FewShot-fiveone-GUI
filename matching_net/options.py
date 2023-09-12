import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-root', '--dataset_root',
        type=str,
        help='Path to dataset, default="../data"',
        default='..' + os.sep + 'data'
    )
    
    parser.add_argument(
        '-name', '--dataset-name',
        type=str,
        help='The name of dataset used for experiment, default=Omniglot. [Omniglot | mini_Imagenet]',
        default='Omniglot'
    )
    
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        help='Learning rate for training, default=0.001',
        default=1e-3
    )
    
    parser.add_argument(
        '-nep', '--n_epochs',
        type=int,
        help='The number of epochs for training, default=100',
        default=100
    )
    
    parser.add_argument(
        '-iter', '--iterations',
        type=int,
        help='The total iterations(eposides) per epoch for training, default=100',
        default=100
    )
    
    parser.add_argument(
        '-lrS', '--lr_scheduler_policy',
        type=str,
        help='learning rate policy, default="step". [step | plateau | cosine]',
        default='step'
    )
    
    parser.add_argument(
        '-lrD', '--lr_decay_epochs',
        type=int,
        help='Multiply by a gamma every lr_decay_epochs epcochs, default=20',
        default=20
    )
    
    parser.add_argument(
        '-lrG', '--lr_gamma',
        type=float,
        help='StepLR learning rate scheduler gamma, default=0.5',
        default=0.5
    )
    
    parser.add_argument(
        '-cTr', '--n_classes_per_iter_training',
        type=int,
        help='The number of random classes per iteration (eposide) for traning dataset, default=60',
        default=60
    )
    
    parser.add_argument(
        '-cTs', '--n_classes_per_iter_test',
        type=int,
        help='The number of random classes per iteration (eposide) for test dataset, default=5',
        default=5
    )
    
    parser.add_argument(
        '-sTr', '--support_samples_per_class_training',
        type=int,
        help='The number of random samples of support set per class for training dataset, default=1',
        default=1
    )
    
    parser.add_argument(
        '-qTr', '--query_samples_per_class_training',
        type=int,
        help='The number of random samples of query set per class for training dataset, default=1',
        default=1
    )
    
    parser.add_argument(
        '-sTs', '--support_samples_per_class_test',
        type=int,
        help='The number of random samples of support set per classes for test dataset, default=1',
        default=1
    )
    
    parser.add_argument(
        '-qTs', '--query_samples_per_class_test',
        type=int,
        help='The number of random samples of query set per classes for test dataset, default=1',
        default=1
    )
    
    parser.add_argument(
        '-gpu', '--gpu_ids',
        type=str,
        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU',
        default='0'
    )
    
    parser.add_argument(
        '-d', '--distance_fn',
        type=str,
        help='The distance function used for measuring distance between query set and prototypical center, [euclidean | cosine]',
        default='euclidean'
    )
    
    parser.add_argument(
        '-pf', '--print_freq',
        type=int,
        help='Printing training and test results on console every "print_freq" epochs',
        default=1
    )
    
    parser.add_argument(
        '-sd', '--save_dir',
        type=str,
        help='Model is saved here, default="../checkpoints"',
        default='..' + os.sep + 'checkpoints',
    )
    
    parser.add_argument(
        '-seed', '--manual-seed',
        type=int,
        help='Input for the manual seeds initializations, default=520, -1 for not fixing seed',
        default=-1 
    )
    
    parser.add_argument(
        '-dl', '--download',
        type=int,
        help='Whether download dataset, default=1. [1 | 0]',
        default=1
    )
    
    parser.add_argument(
        '-hd', '--n_hiddens',
        type=int,
        help='The number of filters of hidden conv layer for protonet, default=64',
        default=64
    )
    
    parser.add_argument(
        '-oc', '--out_channels',
        type=int,
        help='The number of filters of last conv layer for protonet, default=64',
        default=64
    )
    
    parser.add_argument(
        '-init', '--init_type',
        type=str,
        help='Network initialization method, default=normal. [normal | xavier | kaiming | orthogonal]',
        default='normal'
    )
    
    parser.add_argument(
        '-gain', '--init_gain',
        type=float,
        help='Scaling factor for normal, xavier and orthogonal, default=0.02',
        default=0.02
    )
    
    parser.add_argument(
        '-fce', '--use_fce',
        type=int,
        help='Whether or not use full context embedding (FCE) in Matching Network, default=1. [1 | 0]',
        default=1
    )
    
    parser.add_argument(
        '-pd', '--plot_dir',
        type=str,
        help='Path to store plotting, default=../plot',
        default='..' + os.sep + 'plot'
    )
    
    return parser