from loss_functions.ConvKernelGenerator import generate_custom_kernel

import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalMSELossIOU(nn.Module):
    def __init__(self, kernel_size=3, attenuation=1, scaling_factor=1, device='cuda'):
        super(ConvolutionalMSELossIOU, self).__init__()
        self.kernel_size = kernel_size
        # Generate and configure the custom kernel on the specified device
        self.kernel = generate_custom_kernel(kernel_size, attenuation, scaling_factor, device)
        self.kernel = self.kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        # Register the kernel as a parameter without gradients
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)

    def IOU_score(self, outputs, labels):
        outputs = outputs.view(-1)
        labels = labels.view(-1)

        intersection = (outputs * labels).sum()
        union = outputs.sum() + labels.sum() - intersection

        iou = intersection / (union + 1e-6)  # Add a small epsilon for numerical stability

        return iou


    def forward(self, prediction, label):
        # Ensure inputs and targets have a channel dimension
        prediction = prediction.unsqueeze(1)  # Shape: [B, 1, H, W]
        label = label.unsqueeze(1)  # Shape: [B, 1, H, W]

        iou = self.IOU_score(prediction, label)

        # Compute the squared difference
        diff = prediction - label
        squared_diff = diff ** 2

        # Apply convolution to the squared differences
        convolved_errors = F.conv2d(squared_diff, self.kernel, padding=self.kernel_size // 2)
        scaled_convolved_errors = convolved_errors / iou

        # Compute the mean of the convolved errors
        loss = scaled_convolved_errors.mean()
        return loss
