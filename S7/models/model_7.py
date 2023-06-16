"""
TARGET:
    - Increase the number of parameters
RESULT:
    - Parameters: 11,994
    - Best training accuracy: 99.29%
    - Best testing accuracy: 99.33%
ANALYSIS:
    - The accuracy is oscillating around 99.2%
    - The model can never reach 99.4% accuracy
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_7(mc.Model_Composite):
    def __init__(self):
        super(Net_7, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=0, bias=False),
            nn.BatchNorm2d(10),
                nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(10, 10, 3, padding=0, bias=False),
            nn.BatchNorm2d(10),
                nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(10, 20, 3, padding=0, bias=False),
            nn.BatchNorm2d(20),
                nn.ReLU(),
            nn.Dropout(0.05)
        ) # Receptive Field: 7

        self.pool1 = nn.MaxPool2d(2, 2)  # Receptive Field: 8

        self.convblock2 = nn.Sequential(
            nn.Conv2d(20, 10, 1, padding=0, bias=False),
            nn.BatchNorm2d(10),
                nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(10, 10, 3, padding=0, bias=False),
            nn.BatchNorm2d(10),
                nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(10, 20, 3, padding=0, bias=False),
            nn.BatchNorm2d(20),
                nn.ReLU(),
            nn.Dropout(0.05)
        ) # Receptive Field: 16

        self.convblock3 = nn.Sequential(
            nn.Conv2d(20, 32, 3, padding=0, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            nn.Conv2d(32, 10, 1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
        ) # Receptive Field: 32

    def forward(self, x):
        x = self.pool1(self.convblock1(x))
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
