"""
TARGET:
    - Add batch normalization to increase train and test accuracy
RESULT:
    - Parameters: 10,970
    - Best training accuracy: 99.88%
    - Best testing accuracy: 99.27%
ANALYSIS:
    - Improved accuracy compared to previous model
    - Model starts overfitting
    - The gap between train and test accuracy is increased
    - Model will not be able to achieve 99.4% accuracy
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_4(mc.Model_Composite):
    def __init__(self):
        super(Net_4, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=0, bias=False),
            nn.BatchNorm2d(10),
                nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=0, bias=False),
            nn.BatchNorm2d(10),
                nn.ReLU(),
            nn.Conv2d(10, 20, 3, padding=0, bias=False),
            nn.BatchNorm2d(20),
                nn.ReLU()
        ) # Receptive Field: 7

        self.pool1 = nn.MaxPool2d(2, 2)  # Receptive Field: 8

        self.convblock2 = nn.Sequential(
            nn.Conv2d(20, 10, 1, padding=0, bias=False),
            nn.BatchNorm2d(10),
                nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=0, bias=False),
            nn.BatchNorm2d(10),
                nn.ReLU(),
            nn.Conv2d(10, 20, 3, padding=0, bias=False),
            nn.BatchNorm2d(20),
                nn.ReLU()
        ) # Receptive Field: 16

        self.convblock3 = nn.Sequential(
            nn.Conv2d(20, 10, 1, padding=0, bias=False),
            nn.BatchNorm2d(10),
                nn.ReLU(),
            nn.Conv2d(10, 10, 7, padding=0, bias=False)
        ) # Receptive Field: 32

    def forward(self, x):
        x = self.pool1(self.convblock1(x))
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
