import torch
from torch import nn


class Hybrid_Bottleneck(nn.Module):
    def __init__(self, input_C, input_L, num_conv2D,
                 channel_ratio=2, kernel_size=3, stride=1, padding=1, bias=False, swin=None):
        super(Hybrid_Bottleneck, self).__init__()

        self.swin = swin

        self.conv2D_layers = Hybrid_Bottleneck.make_conv2D_layers(in_channels=input_C,
                                                                  num_conv2D=num_conv2D,
                                                                  channel_ratio=channel_ratio,
                                                                  kernel_size=kernel_size,
                                                                  stride=stride,
                                                                  padding=padding,
                                                                  bias=bias)

        flatten_length = input_C * (channel_ratio ** num_conv2D)
        flatten_length_with_1d = input_C * (channel_ratio ** num_conv2D) + input_L

        # downsample the channel
        self.downSampling_block = nn.Sequential(
            nn.Conv2d(in_channels=flatten_length_with_1d,
                      out_channels=flatten_length,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=bias),
            nn.BatchNorm2d(flatten_length),
            nn.ReLU(inplace=True)
        )

        self.deconv2D_layers = Hybrid_Bottleneck.make_deconv2D_layers(
            in_channels=input_C * (channel_ratio ** num_conv2D),
            num_conv2D=num_conv2D,
            channel_ratio=channel_ratio,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)

    def forward(self, x_2d, x_1d):
        # Flatten the 2D input to (B, C, H*W)

        if self.swin is not None:
            x_2d_reshape = x_2d.clone()
            x_2d_reshape = x_2d_reshape.view(x_2d_reshape.shape[0], self.swin[0], self.swin[1], x_2d_reshape.shape[2]).permute(0, 3, 1, 2).contiguous()
            inter_x = self.conv2D_layers(x_2d_reshape)

            concatenated = torch.cat((inter_x, x_1d.view(x_1d.size(0), -1, 1, 1)),
                                     dim=1)  # Resulting shape: (B, C + L, 1,1)

            down_sampled_concatenated = self.downSampling_block(concatenated)

            output = self.deconv2D_layers(down_sampled_concatenated)
            output = output.permute(0, 2, 3, 1).contiguous().view(output.shape[0], self.swin[0] * self.swin[1], -1)

        else:
            inter_x = self.conv2D_layers(x_2d)

            concatenated = torch.cat((inter_x, x_1d.view(x_1d.size(0), -1, 1, 1)),
                                     dim=1)  # Resulting shape: (B, C + L, 1,1)

            down_sampled_concatenated = self.downSampling_block(concatenated)

            output = self.deconv2D_layers(down_sampled_concatenated)

        return output + x_2d

    @staticmethod
    def make_conv2D_layers(in_channels, num_conv2D, channel_ratio, kernel_size, stride, padding, bias):
        layers = []
        in_ch = in_channels
        out_ch = in_channels * channel_ratio

        for i in range(num_conv2D):
            conv2D_layer = nn.Conv2d(in_channels=in_ch,
                                     out_channels=out_ch,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias)
            layers.append(conv2D_layer)

            batchnorm2D_layer = nn.BatchNorm2d(out_ch)
            layers.append(batchnorm2D_layer)

            relu_layer = nn.ReLU(inplace=True)
            layers.append(relu_layer)

            # Update in_channels for the next layer
            in_ch = in_ch * channel_ratio
            out_ch = out_ch * channel_ratio

        return nn.Sequential(*layers)

    @staticmethod
    def make_deconv2D_layers(in_channels, num_conv2D, channel_ratio, kernel_size, stride, padding, bias):
        layers = []
        in_ch = in_channels
        out_ch = in_channels // channel_ratio

        for i in range(num_conv2D):
            conv2D_layer = nn.ConvTranspose2d(in_channels=in_ch,
                                              out_channels=out_ch,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              bias=bias)
            layers.append(conv2D_layer)

            batchnorm2D_layer = nn.BatchNorm2d(out_ch)
            layers.append(batchnorm2D_layer)

            relu_layer = nn.ReLU(inplace=True)
            layers.append(relu_layer)

            # Update in_channels for the next layer
            in_ch = in_ch // channel_ratio
            out_ch = out_ch // channel_ratio

        return nn.Sequential(*layers)
