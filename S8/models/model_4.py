"""
TARGET:
    - Add dropout to reduce overfitting for the same model
RESULT:
    - Parameters: 30,552
    - Best training accuracy: 75.41%
    - Best testing accuracy: 75.43%
ANALYSIS:
    - It can be noted that the overfitting is addressed with dropout layers
    - The gap between train and test accuracy is reduced
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_4(mc.Model_Composite):
    def __init__(self, in_ch):
        super(Net_4, self).__init__()
        dropout_val = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 5

        self.trans1 = nn.Sequential(
            nn.Conv2d(8, 8, 1, padding=0, bias=False),
        ) # Receptive Field: 6
        
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 18

        self.trans2 = nn.Sequential(
            nn.Conv2d(16, 16, 1, padding=0, bias=False),
        ) # Receptive Field: 20

        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 44

        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 10, 1, padding=0, bias=False)
        ) # Receptive Field: 72

    def forward(self, x):
        x = self.convblock1(x)
        identity = self.trans1(x)
        x = x.clone() + identity
        x = self.pool1(x)

        x = self.convblock2(x)
        identity = self.trans2(x)
        x = x.clone() + identity
        x = self.pool2(x)

        x = self.convblock3(x)
        x = self.last(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
