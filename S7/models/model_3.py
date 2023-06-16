"""
TARGET:
    - Reduce the number of parameters from the previous model without changing the skeleton
RESULT:
    - Parameters: 10,790
    - Best training accuracy: 99.21%
    - Best testing accuracy: 98.77%
ANALYSIS:
    - The model is far better than model 1 since the gap between training and testing accuracy is less with far less parameters.
    - Model is overfitting after 11 epochs
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_3(mc.Model_Composite):
    def __init__(self):
        super(Net_3, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=0, bias=False),
                nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=0, bias=False),
                nn.ReLU(),
            nn.Conv2d(10, 20, 3, padding=0, bias=False),
                nn.ReLU()
        ) # Receptive Field: 7

        self.pool1 = nn.MaxPool2d(2, 2)  # Receptive Field: 8

        self.convblock2 = nn.Sequential(
            nn.Conv2d(20, 10, 1, padding=0, bias=False),
                nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=0, bias=False),
                nn.ReLU(),
            nn.Conv2d(10, 20, 3, padding=0, bias=False),
                nn.ReLU()
        ) # Receptive Field: 16

        self.convblock3 = nn.Sequential(
            nn.Conv2d(20, 10, 1, padding=0, bias=False),
                nn.ReLU(),
            nn.Conv2d(10, 10, 7, padding=0, bias=False)
        ) # Receptive Field: 32

    def forward(self, x):
        x = self.pool1(self.convblock1(x))
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
