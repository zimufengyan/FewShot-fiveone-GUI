import numpy as np
import torch 

class OneShotSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.all_labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        self.classes = np.array(list(set(self.all_labels)))
        
    def sample_batch(self, n_ways):
        """Sample a batch of images, size of (n_ways + 1, ...)"""
        support_classes = np.random.choice(self.classes, n_ways)
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
    
    def transformer(self, samples, transform):
        if isinstance(samples, list):
            ret = []
            for sample in samples:
                ret.append(transform(sample))
            ret = torch.stack(ret)
        else:
            ret = transform(samples)
        return ret
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import datasets
    from torchvision.transforms import transforms, InterpolationMode
    from torchvision.utils import make_grid
    
    transform = transforms.Compose([
        transforms.Resize((28, 28), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.87, std=0.33)
    ])
    test_dataset = datasets.Omniglot(root='./data', background=False, transform=transform, download=False)
    sampler = OneShotSampler(test_dataset)
    samples, labels = sampler.sample_batch(5)
    print(samples.size())
    
    plt.figure()
    plt.axis("off")
    plt.imshow(np.transpose(make_grid(samples, padding=2, normalize=True), (1,2,0)))
    plt.show()