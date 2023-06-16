"""
TARGET:
    - Replace conv 7x7 by GAP
RESULT:
    - Parameters: 6,050
    - Best training accuracy: 98.59%
    - Best testing accuracy: 99%
ANALYSIS:
    - Because of the reduction in the number of parameters, the accuracy is reduced for both training and testing
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_6(mc.Model_Composite):
    def __init__(self):
        super(Net_6, self).__init__()
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
            nn.Conv2d(20, 10, 1, padding=0, bias=False),
            # nn.BatchNorm2d(10),
            #     nn.ReLU(),
            # nn.Dropout(0.05),
            nn.AvgPool2d(7)
        ) # Receptive Field: 32

    def forward(self, x):
        x = self.pool1(self.convblock1(x))
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
