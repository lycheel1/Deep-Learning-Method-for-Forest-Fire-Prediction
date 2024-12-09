# 11：based on 8, the multi-head starts from QKV instead of input of the attn block

import torch
import torch.nn as nn

from NN_models.models_wf02.embedding_block import Embedding_block


class Custom_ASPCUNet11_only2D(nn.Module):

    def __init__(self, in_channels=(5 + 8 - 1), out_channels=1, init_features=32, num_1d_features=51):
        super(Custom_ASPCUNet11_only2D, self).__init__()

        features = init_features

        self.fuel_embedding = Embedding_block(embedding_dim=8)

        self.encoder1 = ASPC_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ASPC_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ASPC_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ASPC_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Bottleneck_block(features * 8, features * 16, num_1d_features)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.attn4 = attention_layer(features * 8, features * 8, attn_window_size=2+1, head_num=8, norm_shape=(8,8))
        self.decoder4 = ASPC_block((features * 8) * 1, features * 8)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.attn3 = attention_layer(features * 4, features * 4, attn_window_size=4+1, head_num=4, norm_shape=(16,16))
        self.decoder3 = ASPC_block((features * 4) * 1, features * 4)

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.attn2 = attention_layer(features * 2, features * 2, attn_window_size=6+1, head_num=2, norm_shape=(32,32))
        self.decoder2 = ASPC_block((features * 2) * 1, features * 2)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.attn1 = attention_layer(features, features, attn_window_size=8+1, head_num=1, norm_shape=(64,64))
        self.decoder1 = ASPC_block(features * 1, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x_2d_in, x_1d):
        x_2d = self.fuel_embedding(x_2d_in)

        enc1 = self.encoder1(x_2d)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4), x_1d)

        dec4 = self.upconv4(bottleneck)
        dec4 = self.attn4(dec4, enc4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.attn3(dec3, enc3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.attn2(dec2, enc2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.attn1(dec1, enc1)
        dec1 = self.decoder1(dec1)

        output = self.conv(dec1)
        output = torch.sigmoid(output)
        output = torch.squeeze(output, 1)
        return output


class ASPC_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPC_block, self).__init__()

        self.channel_adjustment_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                                     nn.BatchNorm2d(num_features=out_channels),
                                                     nn.ReLU(inplace=True))

        self.conv_1x1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True))

        self.conv_3x3_dil1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(num_features=out_channels),
                                           nn.ReLU(inplace=True))

        self.conv_3x3_dil2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
                                           nn.BatchNorm2d(num_features=out_channels),
                                           nn.ReLU(inplace=True))

        self.conv_3x3_dil3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3),
                                           nn.BatchNorm2d(num_features=out_channels),
                                           nn.ReLU(inplace=True))

        self.compression = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
                                         nn.BatchNorm2d(num_features=out_channels),
                                         nn.ReLU(inplace=True))

        # self.residue_conv_1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #                                       nn.BatchNorm2d(num_features=out_channels),
        #                                       nn.ReLU(inplace=True))

    def forward(self, x):
        x0 = self.channel_adjustment_conv(x)

        x1 = self.conv_1x1(x0)
        x2 = self.conv_3x3_dil1(x0)
        x3 = self.conv_3x3_dil2(x0)
        x4 = self.conv_3x3_dil3(x0)

        # Concatenate along the channel dimension
        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.compression(out)

        return out + x0


class Bottleneck_block(nn.Module):
    def __init__(self, in_channels, features, num_1d_features):
        super(Bottleneck_block, self).__init__()

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                   out_channels=features,
                                                   kernel_size=3,
                                                   padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(num_features=features),
                                         nn.ReLU(inplace=True))
        #
        # self.bottleneck = Hybrid_Bottleneck(input_C=features, input_L=num_1d_features, num_conv2D=1,
        #                                     channel_ratio=4, kernel_size=4, stride=1, padding=0)

        self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels=features,
                                                   out_channels=features,
                                                   kernel_size=3,
                                                   padding=1,
                                                   bias=False),
                                         nn.BatchNorm2d(num_features=features),
                                         nn.ReLU(inplace=True))

    def forward(self, x_2d, x_1d):
        temp = self.conv_block1(x_2d)
        # temp = self.bottleneck(temp, x_1d)
        out = self.conv_block2(temp)

        return out

class attention_layer(nn.Module):
    def __init__(self, in_channels, out_channels, attn_window_size, head_num, norm_shape):
        super(attention_layer, self).__init__()
        self.conv2D_Q = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True))

        self.conv2D_K = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True))

        self.conv2D_V = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True))


        self.conv2D_attn = nn.Sequential(nn.Conv2d(in_channels=out_channels * 2, out_channels=head_num,
                                                   kernel_size=attn_window_size, stride=1,
                                                   padding='same', padding_mode= 'zeros', bias=False, groups=head_num),
                                         # nn.BatchNorm2d(num_features=out_channels),
                                         # nn.ReLU(inplace=True),
                                         nn.Softmax(dim=1))

        self.expand_times = out_channels//head_num
        self.norm = nn.LayerNorm(normalized_shape=norm_shape)


    def forward(self, x_Q, x_KV):
        Q = self.conv2D_Q(x_Q)
        K = self.conv2D_K(x_KV)
        V = self.conv2D_V(x_KV)

        QK = torch.stack((Q, K), dim=2) # (B,C,2,H,W)
        QK_interleave = QK.view(QK.shape[0], QK.shape[1]*2, QK.shape[-2], QK.shape[-1])

        attn_score = self.conv2D_attn(QK_interleave)
        attn_score_expanded = attn_score.repeat_interleave(self.expand_times, dim=1)
        attn = torch.mul(attn_score_expanded, V)
        out = self.norm(attn) + x_Q

        return out




class attention_layer_backup(nn.Module):
    def __init__(self, in_channels, out_channels, attn_window_size):
        super(attention_layer, self).__init__()
        self.conv2D_Q = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True))

        self.conv2D_K = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True))

        self.conv2D_V = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True))


        self.conv2D_attn = nn.Sequential(nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels,
                                                   kernel_size=attn_window_size, stride=1,
                                                   padding='same', padding_mode= 'circular', bias=False),
                                         # nn.BatchNorm2d(num_features=out_channels),
                                         # nn.ReLU(inplace=True),
                                         nn.Softmax(dim=1))


    def forward(self, x_Q, x_KV):
        Q = self.conv2D_Q(x_Q)
        K = self.conv2D_K(x_KV)
        V = self.conv2D_V(x_KV)

        attn_score = self.conv2D_attn(torch.cat([Q, K], dim=1))
        attn = torch.mul(attn_score, V)
        return attn

