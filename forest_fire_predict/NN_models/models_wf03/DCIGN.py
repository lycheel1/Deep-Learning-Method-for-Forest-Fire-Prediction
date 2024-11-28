import torch
import torch.nn as nn

from NN_models.models_wf02.embedding_block import Embedding_block
import torch.nn.functional as F


class custom_DCIGN(nn.Module):
    def __init__(self, in_channels=(5+8-1), num_of_1d=51):
        super(custom_DCIGN, self).__init__()

        self.fuel_embedding = Embedding_block(embedding_dim=8)

        self.input_dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=10, stride=1, padding='same')
        self.conv1_relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=10, stride=1, padding='same')
        self.conv2_relu = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mlp1 = nn.Linear(in_features=256*64+num_of_1d, out_features=64*64*2)
        self.tanh = nn.Tanh()

        self.deconv1 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=10, stride=1, padding=5)

    def forward(self, x_2d_in, x_1d):
        x_2d = self.fuel_embedding(x_2d_in)

        x_2d = self.input_dropout(x_2d)

        x_2d = self.conv1(x_2d)
        x_2d = self.conv1_relu(x_2d)
        x_2d = self.maxpool1(x_2d)

        x_2d = self.conv2(x_2d)
        x_2d = self.conv2_relu(x_2d)
        x_2d = self.maxpool2(x_2d)


        x_2d_flatten = torch.flatten(x_2d, start_dim=1)
        x_mix = torch.cat((x_2d_flatten, x_1d), dim=1) # (B, C*H*W) + (B, L)

        x_mix = self.mlp1(x_mix)
        x_mix = self.tanh(x_mix)

        x_mix = x_mix.view(x_mix.shape[0], 2, 64, 64)
        x_mix = F.pad(x_mix, (0, 1, 0, 1), "constant", 0)
        x_mix = self.deconv1(x_mix)

        return torch.squeeze(F.sigmoid(x_mix), 1)






