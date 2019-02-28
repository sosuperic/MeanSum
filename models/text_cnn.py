# text_cnn.py

"""
Simple CNN model that can be used for classification of text
"""

import math
import pdb

import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class BasicTextCNN(nn.Module):
    def __init__(self, filter_sizes, n_feat_maps, emb_size, dropout_prob):
        """
        Args:
            filter_sizes: list of ints
                - Size of convolution window (referred to as filter widths in original paper)
            n_feat_maps: int
                - Number of output feature maps for each filter size
            emb_size: int
                - Size of word embeddings the model operates over
            dropout_prob: float
        """
        super(BasicTextCNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.n_feat_maps = n_feat_maps

        self.act = nn.ReLU()
        self.cnn_modlist = nn.ModuleList(
            [nn.Conv2d(1, n_feat_maps, (filter_size, emb_size)) for filter_size in filter_sizes])
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, emb_size]

        Returns:

        """
        x = x.unsqueeze(1)  # [batch, 1, seq_len, emb_size]

        cnn_relus = [self.act(cnn(x)) for cnn in self.cnn_modlist]
        # Each is [batch, n_feat_maps, seq_len-filter_size+1, 1]

        # Pool over time dimension
        pooled = [F.max_pool2d(cnn_relu, (cnn_relu.size(2), 1)).squeeze(3).squeeze(2) for cnn_relu in cnn_relus]
        # Each is [batch, n_feat_maps]

        outputs = T.cat(pooled, 1)  # [batch, n_feat_maps * len(filter_sizes)]
        outputs = self.dropout(outputs)

        return outputs

if __name__ == '__main__':
    cnn = BasicTextCNN([3,4,5], 128, 256, 0.1)
    x = torch.rand(8, 30, 256)
    cnn(x)