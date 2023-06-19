import torch
import torch.nn as nn
from torch.nn import functional as F

class CiteModel(nn.Module):
    """Makotu original model"""

    def __init__(self, feature_num):
        super(CiteModel, self).__init__()

        self.layer_seq_256 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.ReLU(),
                                      )
        self.layer_seq_64 = nn.Sequential(nn.Linear(128, 64),
                                       nn.Linear(64, 32),
                                       nn.LayerNorm(32),
                                       nn.ReLU(),
                                      )
        self.layer_seq_8 = nn.Sequential(nn.Linear(32, 16),
                                         nn.Linear(16, 8),
                                       nn.LayerNorm(8),
                                       nn.ReLU(),
                                      )
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(128 + 32 + 8, 140)

    def forward(self, X, y=None):

        X_256 = self.layer_seq_256(X)
        X_64 = self.layer_seq_64(X_256)
        X_8 = self.layer_seq_8(X_64)

        X = torch.cat([X_256, X_64, X_8], axis = 1)
        out = self.head(X)

        return out


class CiteModel_mish(nn.Module):
    """Makotu original model with Mish activation function"""

    def __init__(self, feature_num):
        super(CiteModel_mish, self).__init__()

        self.layer_seq_256 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.Mish(),
                                      )
        self.layer_seq_64 = nn.Sequential(nn.Linear(128, 64),
                                       nn.Linear(64, 32),
                                       nn.LayerNorm(32),
                                       nn.Mish(),
                                      )
        self.layer_seq_8 = nn.Sequential(nn.Linear(32, 16),
                                         nn.Linear(16, 8),
                                       nn.LayerNorm(8),
                                       nn.Mish(),
                                      )

        self.head = nn.Linear(128 + 32 + 8, 140)

    def forward(self, X, y=None):

        X_256 = self.layer_seq_256(X)
        X_64 = self.layer_seq_64(X_256)
        X_8 = self.layer_seq_8(X_64)

        X = torch.cat([X_256, X_64, X_8], axis = 1)
        out = self.head(X)

        return out
