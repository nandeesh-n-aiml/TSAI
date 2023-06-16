"""
TARGET:
    - Slightly reduce the parameters to achieve 99.4% test accuracy
RESULT:
    - Parameters: 11,492
    - Best training accuracy: 99.24%
    - Best testing accuracy: 99.34%
ANALYSIS:
    - The model is not over-fitting
    - The model has potential to learn
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_10(mc.Model_Composite):
    def __init__(self):
        super(Net_10, self).__init__()
        dropout_val = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=0, bias=False),
            nn.BatchNorm2d(8),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(8, 16, 3, padding=0, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 5

        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 10, 1, padding=0, bias=False),
            nn.BatchNorm2d(10),
                nn.ReLU()
        ) # Receptive Field: 6

        self.convblock2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(16, 16, 3, padding=0, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(16, 16, 3, padding=0, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 18

        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16, 10, 1, padding=0, bias=False)
        ) # Receptive Field: 28

    def forward(self, x):
        x = self.convblock1(x)
        x = self.trans1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
