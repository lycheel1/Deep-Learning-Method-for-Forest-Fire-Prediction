import torch
import torch.nn as nn

import torch.nn.functional as F


class custom_MLP1(nn.Module):
    def __init__(self, num_1d_features=51):
        super(custom_MLP1, self).__init__()

        self.MLP1 = nn.Sequential(nn.Linear(in_features=5*64*64+num_1d_features, out_features=128),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(p=0.5))

        self.MLP2 = nn.Sequential(nn.Linear(in_features=128, out_features=64),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(p=0.5))

        self.MLP3 = nn.Sequential(nn.Linear(in_features=64, out_features=32),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(p=0.5))

        self.output_layer = nn.Linear(in_features=32, out_features=64*64)

    def forward(self, x_2d_in, x_1d):
        x_2d = torch.flatten(x_2d_in, start_dim=1)
        x = torch.cat((x_2d, x_1d), dim=1)
        x = self.MLP1(x)
        x = self.MLP2(x)
        x = self.MLP3(x)
        x = self.output_layer(x)
        x = x.view(x.shape[0], 64, 64)

        return F.sigmoid(x)


