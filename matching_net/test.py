import torch

from networks import BidirectionLSTMEmbedding
from base import init_weights

lstm = BidirectionLSTMEmbedding(
    10, [32], True
)
print(lstm.fce.weight_hh_l[0])