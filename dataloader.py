from typing import Any
import numpy as np
import torch 
import os
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose, InterpolationMode
from PIL import ImageOps
import logging


class DatasetLoader:
    def __init__(self):
        logging.basicConfig(
            level = logging.DEBUG,
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.log = logging.getLogger(__name__)
                
        self.dataset = None
        self.sampler = None
        self.transform = None
        self.data_dir = '.' + os.sep + 'data'
        
    def init_dataset(self, dataset_name):
        if self.dataset is not None and self.dataset.__class__.__name__ == dataset_name:
            self.log.debug("The dataset had been loaded, skipping this operation")
            return
        if dataset_name == 'Omniglot':
            self.log.debug("Loading Omniglot dataset")
            # When training model I use torchmeta to load dataset, and the background is black and charactor is white in each sample. 
            # But in this case, I use torchvision to load dataset, and the background is white and charactor is black in each sample. 
            # So convert the color in each sample here
            self.transform = Compose([
                ConvertColor(),
                Resize(28, interpolation=InterpolationMode.BICUBIC), 
                ToTensor()
            ])
            # use transform before prediction, but after showing
            self.dataset = datasets.Omniglot(
                root=self.data_dir, background=False, 
                download=False
            )
            self.sampler = OneShotSampler(self.dataset)
        elif self.dataset_name == 'mini_ImageNet':
            raise  NotADirectoryError
        else:
            raise ValueError(f"Unknown dataset name ({dataset_name})")
        self.log.debug("Dataset has been loaded")
        
    def get_batch(self, n_way):
        """Get a batch of n-way-1-shot support set, 1 pair of query sampele and label"""
        return self.sampler.sample_batch(n_way)
    
    def transformer(self, samples):
        # transfomer given samples with type <PIL.Image.Image>
        if isinstance(samples, list):
            ret = []
            for sample in samples:
                ret.append(self.transform(sample))
            ret = torch.stack(ret)
        else:
            ret = self.transform(samples).unsqueeze(0)
        return ret
    
    def is_available(self):
        if self.dataset is None or self.sampler is None:
            return False
        else :
            return True
    
    
class OneShotSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.all_labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        self.classes = np.array(list(set(self.all_labels)))
        
    def sample_batch(self, n_way):
        """Sample a batch of images, size of (n_way + 1, ...)"""
        support_classes = np.random.choice(self.classes, n_way)
        query_class_idx = np.random.randint(0, len(support_classes))
        query_class = support_classes[query_class_idx]
        
        support_set = []
        query = None
        
        for c in support_classes:
            idxs = np.argwhere(self.all_labels == c)
            if c == query_class:
                pair_idxs = np.random.choice(idxs.flatten(), 2)
                support_idx = pair_idxs[0]
                target_idx = pair_idxs[1]
                query = self.dataset[target_idx][0]
            else:
                support_idx = np.random.choice(idxs.flatten(), 1).item()
            support_set.append(self.dataset[support_idx][0])
            
            
        return support_set, query, query_class_idx
    

class ConvertColor:
    """Reverse the black and white of the input image"""
    def __init__(self) -> None:
        pass

    def __call__(self, img) -> Any:
        return ImageOps.invert(img)