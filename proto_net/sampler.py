import numpy as np
import torch


class FewShotBatchSampler:
    """Defien a batch sampler for few shot task"""
    def __init__(self, labels, n_ways, k_shots, iterations) -> None:
        self.labels = labels
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.iterations = iterations
        
        self.classes, self.count = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)
        
        # create a matrix, indexes, of dim: (self.classes, max(self.count))
        # fill it with zeros.
        self.idxs = range(len(self.labels))        
        self.indexes = np.zeros(shape=(len(self.classes), max(self.count)), dtype=np.int)
        self.numel_per_class = torch.zeros_like(self.classes)   # store the number of samples of each class\row
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(self.indexes[label_idx] == 0)[0][0]] = idx
            self.numel_per_class[label_idx] += 1
        
    def __iter__(self):
        """yield a batch of samples"""
        n, k = self.n_ways, self.k_shots
        
        for _ in range(self.iterations):
            b_size = n * k
            batch = np.zeros(b_size, dtype=np.int)
            c_idxs = torch.randperm(len(self.classes))[:n]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * k, (i + 1) * k)
                label_idx = torch.arange(len(self.classes)).long()[c_idxs[i]].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:k]
                batch[s] = self.indexes[label_idx][sample_idxs]
            # batch = np.random.shuffle(batch)
            batch = torch.LongTensor(batch)
            
            yield batch
                
    def __len__(self):
        """returns the number of iterations (episodes) per epoch"""
        return self.iterations
            