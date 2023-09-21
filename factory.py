import logging
import os
import torch

from metric_models.matching import MatchingModel
from metric_models.proto import ProtoModel
from metric_models.relation import RelationModel
from metric_models.selfatten_relation import SelfAttentionRelationModel
from meta_models.maml import FOMAML
from meta_models.networks import Classifier


class ModelFactor:
    def __init__(self):
        logging.basicConfig(
            level = logging.DEBUG,
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.log = logging.getLogger(__name__)
        
        self.model_names = ['Matching Network' ,'Proto Network', 'Relation Network', 'SA Relation Network', 'MAML']
        self.accuracy_dict = {
            'Matching Network': {'Omniglot': 0.9536, },
            'Proto Network': {'Omniglot': 0.9858},
            'Relation Network': {'Omniglot': 0.9752},
            'SA Relation Networ': {'Omniglot': 0.9706},
            'MAML': {'Omniglot': 0.9841},
        }
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
        elif model_name == 'Proto Network':
            model = ProtoModel(input_shape, out_channels, num_hidden=num_hiddens, is_train=False)
        elif model_name == 'Relation Network':
            model = RelationModel(input_shape, num_hiddes=num_hiddens, is_train=False)
        elif model_name == 'SA Relation Network':
            model = SelfAttentionRelationModel(input_shape, num_hiddes=num_hiddens, is_train=False)
        elif model_name == 'MAML':
            learner = Classifier(input_shape, n_classes=n_classes, num_hiddens=num_hiddens)
            model = FOMAML(learner, num_classes=n_classes)
        else:
            raise ValueError(f"Unknown model name ({model_name})")
        
        model.load_networks(self.load_dir, dataset_name, epoch)
        model.to_device()
        return model
    
    def pred(self, model, **kwargs):
        Xs, Xq = kwargs['Xs'], kwargs['Xq']
        if model.__class__.__name__ in ['MAML', 'ModelAgnosticMetaLearning']:
            ys = kwargs.get('ys', torch.arange(0, Xs.size(0)).long())
            return model.pred(Xs, ys, Xq, step=5, lr=0.4).item()
        else:
            n_way = kwargs['n_way']
            return model.pred(Xs, Xq, n_way).item()
    
    def get_accuracy(self, model_name, dataset_name) -> int:
        if self.accuracy_dict.get(model_name, None) is None:
            return 0
        return self.accuracy_dict[model_name].get(dataset_name, 0)


