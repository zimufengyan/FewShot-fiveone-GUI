import logging
import os
import torch

from metric_models.matching import MatchingModel
from metric_models.proto import ProtoModel
from metric_models.relation import RelationModel
from metric_models.selfatten_relation import SelfAttentionRelationModel
from meta_models.maml import FOMAML
from meta_models.meta_sgd import MetaSGD
from meta_models.networks import Classifier
from snail.model import SnailModel


class ModelFactor:
    def __init__(self):
        logging.basicConfig(
            level = logging.DEBUG,
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.log = logging.getLogger(__name__)
        
        self.accuracy_dict = {
            'Matching Network': {'Omniglot': 0.9536},
            'Proto Network': {'Omniglot': 0.9858},
            'Relation Network': {'Omniglot': 0.9752},
            'SA Relation Networ': {'Omniglot': 0.9706},
            'MAML': {'Omniglot': 0.9841},
            'Meta-SGD': {'Omniglot': 0.9834},
            'SNAIL': {'Omniglot': 0.9575},
        }
        self.model_names = self.accuracy_dict.keys()
        
        self.load_dir = '.' + os.sep + 'checkpoints'
        
    def get_model(self, model_name, dataset_name, n_classes=5):
        if dataset_name == 'Omniglot':
            out_channels, num_hiddens = 64, 64
            input_shape = [1, 28, 28]
            epoch = 100
        else:
            raise NotImplementedError
        if model_name == 'Matching Network':
            model = MatchingModel(input_shape, out_channels, num_hiddens=num_hiddens, is_train=False)
            model.load_networks(self.load_dir, dataset_name, epoch)
        elif model_name == 'Proto Network':
            model = ProtoModel(input_shape, out_channels, num_hidden=num_hiddens, is_train=False)
            model.load_networks(self.load_dir, dataset_name, epoch)
        elif model_name == 'Relation Network':
            model = RelationModel(input_shape, num_hiddes=num_hiddens, is_train=False)
            model.load_networks(self.load_dir, dataset_name, epoch)
        elif model_name == 'SA Relation Network':
            model = SelfAttentionRelationModel(input_shape, num_hiddes=num_hiddens, is_train=False)
            model.load_networks(self.load_dir, dataset_name, epoch)
        elif model_name == 'MAML':
            learner = Classifier(input_shape, n_classes=n_classes, num_hiddens=num_hiddens)
            model = FOMAML(learner, num_classes=n_classes, is_train=False)
            model.load_networks(self.load_dir, dataset_name, epoch)
        elif model_name == 'Meta-SGD':
            learner = Classifier(input_shape, n_classes=n_classes, num_hiddens=num_hiddens)
            model = MetaSGD(learner, num_classes=n_classes, is_train=False)
            model.load_networks(self.load_dir, dataset_name, epoch)
        elif model_name == 'SNAIL':
            model = SnailModel(input_shape[0], n_way=n_classes, k_shot=1, dataset_name=dataset_name, 
                               is_train=False)
            model.load_networks(self.load_dir, epoch)
        else:
            raise ValueError(f"Unknown model name ({model_name})")
        
        model.to_device()
        return model
    
    def pred(self, model, **kwargs):
        try:
            Xs, Xq = kwargs['Xs'], kwargs['Xq']
            if model.__class__.__name__ in ['MAML', 'ModelAgnosticMetaLearning']:
                ys = kwargs.get('ys', torch.arange(0, Xs.size(0)).long())
                return model.pred(Xs, ys, Xq, step=5, lr=0.4).item()
            elif model.__class__.__name__ == 'MetaSGD':
                ys = kwargs.get('ys', torch.arange(5, Xs.size(0)).long())
                return model.pred(Xs, ys, Xq, step=3).item()
            elif model.__class__.__name__ == 'SnailModel':
                X = torch.cat([Xs, Xq], dim=0)
                ys = torch.Tensor(list(range(Xs.size(0))))
                y = torch.cat([ys, torch.zeros(1)], dim=0).long()
                y = torch.nn.functional.one_hot(y)
                return model.pred(X, y).item()
            else:
                n_way = kwargs['n_way']
                return model.pred(Xs, Xq, n_way).item()
        except Exception as e:
            print(e)
    
    def get_accuracy(self, model_name, dataset_name) -> int:
        if self.accuracy_dict.get(model_name, None) is None:
            return 0
        return self.accuracy_dict[model_name].get(dataset_name, 0)


