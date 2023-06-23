"""
TARGET:
    - Add skip connections to the previous model
    - Add batch, layer, and group normalization layers separately
    - Achieve a min accuracy of 70%
    - The number of parameters should be less than 50K
RESULT:
    - Parameters: 33,872
    - Batch normalization:
        - Best training accuracy: 78.12%
        - Best testing accuracy: 76.8%
    - Layer normalization:
        - Best training accuracy: 73.81%
        - Best testing accuracy: 71.83%
    - Group normalization:
        - Best training accuracy: 74.82%
        - Best testing accuracy: 73.82%
ANALYSIS:
    - The model with batch normalization layers are having higher accuracy followed by group normalization and layer normalization
    - The gap between train and test accuracy is less in group normalization
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_4(mc.Model_Composite):
    def __init__(self, in_ch, norm_type='bn'):
        super(Net_4, self).__init__()
        dropout_val = 0.1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            self.get_norm(16, type=norm_type),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        self.convblock1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16, type=norm_type),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 5

        self.trans1 = nn.Sequential(
            nn.Conv2d(16, 16, 1, padding=0, bias=False),
        ) # Receptive Field: 6
        
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16, type=norm_type),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16, type=norm_type),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16, type=norm_type),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 18

        self.trans2 = nn.Sequential(
            nn.Conv2d(16, 16, 1, padding=0, bias=False),
        ) # Receptive Field: 20

        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            self.get_norm(32, type=norm_type),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            self.get_norm(32, type=norm_type),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            self.get_norm(32, type=norm_type),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 44

        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 10, 1, padding=0, bias=False)
        ) # Receptive Field: 72

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.convblock1(x)
        x = self.trans1(x)
        x = x.clone() + identity
        x = self.pool1(x)

        x = self.conv2(x)
        identity = x
        x = self.convblock2(x)
        x = self.trans2(x)
        x = x.clone() + identity
        x = self.pool2(x)

        x = self.convblock3(x)
        x = self.last(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
