"""
TARGET:
    - Reduce number of parameters below 50k
    - Test accuracy should be greater than 70%
RESULT:
    - Parameters: 48,528
    - Best training accuracy: 85.78%
    - Best testing accuracy: 76.88%
ANALYSIS:
    - Model is still over-fitting but better than previous model
    - Target achieved
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_2(mc.Model_Composite):
    def __init__(self, in_ch):
        super(Net_2, self).__init__()
        dropout_val = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            # nn.Dropout(dropout_val),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
                nn.ReLU(),
            # nn.Dropout(dropout_val)
        ) # Receptive Field: 5

        self.trans1 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        ) # Receptive Field: 6

        self.convblock2 = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            # nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            # nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            # nn.Dropout(dropout_val)
        ) # Receptive Field: 18

        self.trans2 = nn.Sequential(
            nn.Conv2d(32, 16, 1, padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        ) # Receptive Field: 20

        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            # nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            # nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            # nn.Dropout(dropout_val)
        ) # Receptive Field: 44

        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 10, 1, padding=0, bias=False)
        ) # Receptive Field: 72

    def forward(self, x):
        x = self.convblock1(x)
        x = self.trans1(x)
        x = self.convblock2(x)
        x = self.trans2(x)
        x = self.convblock3(x)
        x = self.last(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
