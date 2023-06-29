"""
TARGET:
    - Add depthwise convolution followed by 1x1 convolution
    - Add dilated conv instead of strided conv
RESULT:
    - Parameters: 171,680
    - Best training accuracy: 76.54%
    - Best testing accuracy: 84.54%
ANALYSIS:
    - The test accuracy is improved compared to the previous model
    - The model is under-fitting due to regularization
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class Net_5(mc.Model_Composite):
    def __init__(self, in_ch, norm_type='bn'):
        super(Net_5, self).__init__()
        dropout_val = 0.05

        self.norm_type = norm_type

        # BLOCK 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), # depth-wise separable conv
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 1, bias=False), # point-wise conv
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, bias=False), # strided conv
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        # BLOCK 2
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), # depth-wise separable conv
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(32, 32, 1, bias=False), # point-wise conv
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, bias=False), # strided conv
            self.get_norm(32),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        # BLOCK 3
        self.conv7 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            self.get_norm(64),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False), # depth-wise separable conv
            self.get_norm(64),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(64, 64, 1, bias=False), # point-wise conv
            self.get_norm(64),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, bias=False), # strided conv
            self.get_norm(64),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        # BLOCK 4
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            self.get_norm(64),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False), # depth-wise separable conv
            self.get_norm(64),
                nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.Conv2d(64, 64, 1, bias=False), # point-wise conv
            self.get_norm(64),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, dilation=2, bias=False), # dilated conv
            self.get_norm(64),
                nn.ReLU(),
            nn.Dropout(dropout_val)
        )

        self.last = nn.Sequential(
            nn.Conv2d(64, 10, 1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        # BLOCK 1
        x = self.conv1(x)
        identity = x
        x = self.conv2(x)
        x = x.clone() + identity
        x = self.conv3(x)
        
        # BLOCK 2
        x = self.conv4(x)
        identity = x
        x = self.conv5(x)
        x = x.clone() + identity
        x = self.conv6(x)
        
        # BLOCK 3
        x = self.conv7(x)
        identity = x
        x = self.conv8(x)
        x = x.clone() + identity
        x = self.conv9(x)
        
        # BLOCK 4
        x = self.conv10(x)
        identity = x
        x = self.conv11(x)
        x = x.clone() + identity
        x = self.conv12(x)
        
        x = self.last(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
