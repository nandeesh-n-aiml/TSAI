"""
TARGET:
    - Model skeleton
    - Use strided convolution instead of max pooling for all convolution blocks
RESULT:
    - Parameters: 63,536
    - Best training accuracy: 83.93%
    - Best testing accuracy: 81.23%
ANALYSIS:
    - Model will not train beyond ~80%
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_1(mc.Model_Composite):
    def __init__(self, in_ch, norm_type='bn'):
        super(Net_1, self).__init__()
        dropout_val = 0.05

        self.norm_type = norm_type

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val),

            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            
            nn.Conv2d(16, 16, 3, stride=2, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 5

        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val),

            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            
            nn.Conv2d(16, 16, 3, stride=2, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            
            nn.Conv2d(32, 32, 3, stride=2, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 10, 1, bias=False)
        ) # Receptive Field: 72

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.last(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
