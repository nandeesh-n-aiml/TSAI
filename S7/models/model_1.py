"""
TARGET:
    - Get the setup right
    - Set data transforms
    - Set data loaders
    - Set basic working code
    - Set basic training and testing loop
RESULT:
    - Parameters: 6,379,786
    - Best training accuracy: 100%
    - Best testing accuracy: 99.35%
ANALYSIS:
    - Heavy model for MNIST dataset
    - Model is overfitting
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_1(mc.Model_Composite):
    def __init__(self):
        super(Net_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.conv7 = nn.Conv2d(1024, 10, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
