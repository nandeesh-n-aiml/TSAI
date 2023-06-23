"""
Session 8 model
TARGET:
    - Add skip connections to the previous model
    - Add batch, layer, and group normalization layers separately
    - Achieve a min accuracy of 70%
    - The number of parameters should be less than 50K
RESULT:
    - Parameters: 33,872
    - Batch normalization:
        - Best training accuracy: 78.12%
        - Best testing accuracy: 76.8%
    - Layer normalization:
        - Best training accuracy: 73.81%
        - Best testing accuracy: 71.83%
    - Group normalization:
        - Best training accuracy: 74.82%
        - Best testing accuracy: 73.82%
ANALYSIS:
    - The model with batch normalization layers are having higher accuracy followed by group normalization and layer normalization
    - The gap between train and test accuracy is less in group normalization
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_4(mc.Model_Composite):
    def __init__(self, in_ch, norm_type='bn', n_groups=2):
        super(Net_4, self).__init__()
        dropout_val = 0.1

        self.norm_type = norm_type
        self.n_groups = n_groups
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        self.convblock1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 5

        self.trans1 = nn.Sequential(
            nn.Conv2d(16, 16, 1, padding=0, bias=False),
        ) # Receptive Field: 6
        
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            self.get_norm(16),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 18

        self.trans2 = nn.Sequential(
            nn.Conv2d(16, 16, 1, padding=0, bias=False),
        ) # Receptive Field: 20

        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        ) # Receptive Field: 44

        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 10, 1, padding=0, bias=False)
        ) # Receptive Field: 72

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.convblock1(x)
        x = self.trans1(x)
        x = x.clone() + identity
        x = self.pool1(x)

        x = self.conv2(x)
        identity = x
        x = self.convblock2(x)
        x = self.trans2(x)
        x = x.clone() + identity
        x = self.pool2(x)

        x = self.convblock3(x)
        x = self.last(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


"""
Session 7 model
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
        ) # Receptive Field: 28

    def forward(self, x):
        x = self.convblock1(x)
        x = self.trans1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


"""
Session 6 model
TARGET:
    - 99.4% validation accuracy
    - Less than 20k Parameters
    - Less than 20 Epochs
RESULT:
    - Parameters: 18,482
    - Best testing accuracy: 99.46%
ANALYSIS:
    - The model has reached 99.4% accuracy within 20 epochs
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
                nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 10, 1),
            nn.AvgPool2d(3)
        )

    def forward(self, x):
        x = self.nn(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)