"""
TARGET:
    - Add transition block when RF is 5
    - Distribute the weight across all the layers
RESULT:
    - Parameters: 13,828
    - Best training accuracy: 99.38%
    - Best testing accuracy: 99.49%
ANALYSIS:
    - The model is not over-fitting
    - The model reached 99.4% accuracy 5 times within 15 epochs but with 13k parameters
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_8(mc.Model_Composite):
    def __init__(self):
        super(Net_8, self).__init__()
        dropout_val = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=0, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(16, 32, 3, padding=0, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 5

        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 10, 1, padding=0, bias=False),
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
        ) # Receptive Field: 32

    def forward(self, x):
        x = self.convblock1(x)
        x = self.trans1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
