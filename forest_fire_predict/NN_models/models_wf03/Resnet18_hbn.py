import torch
import torch.nn as nn
from torchvision import models

from NN_models.models_wf02.embedding_block import Embedding_block
from NN_models.models_wf02.hybrid_bottleneck import Hybrid_Bottleneck


class Custom_ResNet18_HBN(nn.Module):
    def __init__(self, num_1d_features=51):
        super(Custom_ResNet18_HBN, self).__init__()

        self.fuel_embedding = Embedding_block(embedding_dim=8)

        self.initial_conv = nn.Conv2d(5+8-1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        original_resnet = models.resnet18()
        # Keep layers from the first residual block onwards
        self.resnet_layers = nn.Sequential(*list(original_resnet.children())[4:-2])

        self.conv2d_block = nn.Sequential(nn.Conv2d(512, 1024,
                                                    kernel_size=3, stride=2, padding=1, bias=False),
                                          nn.BatchNorm2d(1024),
                                          nn.ReLU(inplace=True))

        # Additional layers for processing concatenated features
        self.fc = nn.Sequential(
            nn.Linear(1024 * (4 ** 2), 64*64),
        )

        self.hybrid_bottleneck1 = Hybrid_Bottleneck(input_C=1024, input_L=num_1d_features, num_conv2D=1,
                                                    channel_ratio=4, kernel_size=4, stride=1, padding=0)


    def forward(self, x_2d_in, x_1d):
        x_2d = self.fuel_embedding(x_2d_in)

        x_2d = self.initial_conv(x_2d)
        x_2d = self.resnet_layers(x_2d)
        x_2d = self.conv2d_block(x_2d)

        # Concatenate the 1D features
        x = self.hybrid_bottleneck1(x_2d, x_1d)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = x.view(x.size(0), -1, 64, 64)

        x = torch.sigmoid(x)
        x = torch.squeeze(x, 1)

        return x



