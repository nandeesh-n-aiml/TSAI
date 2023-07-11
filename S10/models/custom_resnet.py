"""
TARGET:
    - Build a custom ResNet model based on David C. Page model, DAWNBench challenge.
RESULT:
    - Parameters: 6,573,120
    - Best training accuracy: 90.74%
    - Best testing accuracy: 90.15%
ANALYSIS:
    - The gap between train and test accuracies are very low which is indeed a good sign as the model is generalizable.
    - Using the ResNet kind architecture, LRFinder, and OneCycleLR, a testing accuracy of 90% is achieved.
"""

import torch.nn as nn
import torch.nn.functional as F
from . import model_composite as mc

class CustomResNet(mc.Model_Composite):
    """ Custom ResNet model based on David C. Page's model for DAWNBench challenge.
    """
    def __init__(self, in_ch: int, norm_type: str='bn'):
        """ Initialization of the custom ResNet model.

        Args:
            in_ch: input channels
            norm_type: type of normalization
        """
        super(CustomResNet, self).__init__()
        self.norm_type = norm_type

        # Preparation Layer
        self.prep_layer = self.get_conv_bn(in_ch, 64)

        # Layer 1
        self.layer1 = self.get_conv_bn(64, 128, True)
        self.residual1 = self.get_residual(128, 128)

        # Layer 2
        self.layer2 = self.get_conv_bn(128, 256, True)

        # Layer 3
        self.layer3 = self.get_conv_bn(256, 512, True)
        self.residual2 = self.get_residual(512, 512)

        # Classification
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 10, bias=False)

    def get_conv_bn(self, in_ch: int, out_ch: int, pool: bool=False):
        """ Get a convolutional layer with an optional pooling layer

        Args:
            in_ch: input channels
            out_ch: output channels
            pool: flag to indicate of pooling is required
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2) if pool else nn.Sequential(),
            self.get_norm(out_ch),
                nn.ReLU()
        )
    
    def get_residual(self, in_ch: int, out_ch: int):
        """ Get a residual block

        Args:
            in_ch: input channels
            out_ch: output channels
        """
        return nn.Sequential(
            self.get_conv_bn(in_ch, out_ch),
            self.get_conv_bn(in_ch, out_ch)
        )

    def forward(self, x):
        """ Connecting across layers
        """
        x = self.prep_layer(x)

        x = self.layer1(x)
        res1 = self.residual1(x)
        x = x + res1

        x = self.layer2(x)

        x = self.layer3(x)
        res2 = self.residual2(x)
        x = x + res2

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
