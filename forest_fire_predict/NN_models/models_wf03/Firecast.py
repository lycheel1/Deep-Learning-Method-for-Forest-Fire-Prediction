import torch
import torch.nn as nn

from NN_models.models_wf02.embedding_block import Embedding_block
import torch.nn.functional as F


class custom_Firecast(nn.Module):
    def __init__(self, in_channels=5+8-1, num_1d_features=51):
        super(custom_Firecast, self).__init__()

        self.fuel_embedding = Embedding_block(embedding_dim=8)

        self.avgPool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
                                   nn.Sigmoid(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Dropout(p=0.5))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Dropout(p=0.5))

        self.MLP1 = nn.Linear(in_features=2*2*64, out_features=2*2*128, bias=True)
        self.MLP2 = nn.Linear(in_features=2*2*128+num_1d_features, out_features=64*64)

    def forward(self, x_2d_in, x_1d):
        x_2d = self.fuel_embedding(x_2d_in)

        x_2d = self.avgPool1(x_2d)
        x_2d = self.conv1(x_2d)
        x_2d = self.conv2(x_2d)

        x_2d_flatten = torch.flatten(x_2d, start_dim=1)
        x_2d_flatten = self.MLP1(x_2d_flatten)

        x = torch.cat((x_2d_flatten, x_1d), dim=1)
        x = self.MLP2(x)
        x = x.view(-1, 64, 64)

        return F.sigmoid(x)