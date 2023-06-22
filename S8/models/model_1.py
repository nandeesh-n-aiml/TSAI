"""
TARGET:
    - Get the setup right
    - Set data transforms
    - Set data loaders
    - Set basic working code
    - Set basic training and testing loop
RESULT:
    - Parameters: 189,280
    - Best training accuracy: 94.602%
    - Best testing accuracy: 80.93%
ANALYSIS:
    - Model is over-fitting
    - More number of parameters
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_1(mc.Model_Composite):
    def __init__(self, in_ch):
        super(Net_1, self).__init__()
        dropout_val = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            # nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
                nn.ReLU(),
            # nn.Dropout(dropout_val)
        ) # Receptive Field: 5

        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 16, 1, padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        ) # Receptive Field: 6

        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
                nn.ReLU(),
            # nn.Dropout(dropout_val),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
                nn.ReLU(),
            # nn.Dropout(dropout_val),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
                nn.ReLU(),
            # nn.Dropout(dropout_val)
        ) # Receptive Field: 14

        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, padding=0, bias=False),
            nn.MaxPool2d(2, 2)
        ) # Receptive Field: 6

        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
                nn.ReLU(),
            # nn.Dropout(dropout_val),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
                nn.ReLU(),
            # nn.Dropout(dropout_val),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
                nn.ReLU(),
            # nn.Dropout(dropout_val)
        ) # Receptive Field: 14

        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 10, 1, padding=0, bias=False)
        ) # Receptive Field: 28

    def forward(self, x):
        x = self.convblock1(x)
        x = self.trans1(x)
        x = self.convblock2(x)
        x = self.trans2(x)
        x = self.convblock3(x)
        x = self.last(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
