"""
TARGET:
    - Commit to model architecture
    - Calculate the RF at the end of each block
RESULT:
    - Parameters: 194,884
    - Best training accuracy: 99.45%
    - Best testing accuracy: 99.09%
ANALYSIS:
    - The model is still large for MNIST kind of data, but far better than the previous model
    - The gap between the test and train accuracy is very less compared to previous model
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_2(mc.Model_Composite):
    def __init__(self):
        super(Net_2, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=0, bias=False),
                nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=0, bias=False),
                nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=0, bias=False),
                nn.ReLU()
        ) # Receptive Field: 7

        self.pool1 = nn.MaxPool2d(2, 2)  # Receptive Field: 8

        self.convblock2 = nn.Sequential(
            nn.Conv2d(128, 32, 1, padding=0, bias=False),
                nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=0, bias=False),
                nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=0, bias=False),
                nn.ReLU()
        ) # Receptive Field: 16

        self.convblock3 = nn.Sequential(
            nn.Conv2d(128, 10, 1, padding=0, bias=False),
                nn.ReLU(),
            nn.Conv2d(10, 10, 7, padding=0, bias=False)
        ) # Receptive Field: 32

    def forward(self, x):
        x = self.pool1(self.convblock1(x))
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
