from loss_functions.ConvKernelGenerator import generate_custom_kernel

import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalMSELoss(nn.Module):
    def __init__(self, kernel_size=3, attenuation=1, scaling_factor=1, smooth=100, device='cuda'):
        super(ConvolutionalMSELoss, self).__init__()
        self.kernel_size = kernel_size
        # Generate and configure the custom kernel on the specified device
        self.kernel = generate_custom_kernel(kernel_size, attenuation, scaling_factor, device)
        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        # Register the kernel as a parameter without gradients
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)
        self.smoothing_factor = smooth


    def forward(self, prediction, label):
        # Ensure inputs and targets have a channel dimension
        prediction = prediction.unsqueeze(1)  # Shape: [B, 1, H, W]
        label = label.unsqueeze(1)  # Shape: [B, 1, H, W]

        # Compute the squared difference
        diff = prediction - label
        squared_diff = diff ** 2

        # Apply convolution to the squared differences
        convolved_errors = F.conv2d(squared_diff, self.kernel, padding=self.kernel_size // 2)
        filtered_convolved_errors = convolved_errors * squared_diff
        # mask = torch.sigmoid(self.smoothing_factor * (difference - 0.5))

        # Compute the mean of the convolved errors
        loss = filtered_convolved_errors.mean()
        return loss
