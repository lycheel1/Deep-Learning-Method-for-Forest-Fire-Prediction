import torch

def generate_custom_kernel(k, a, scaling_factor=1, device='cuda'):
    if k % 2 == 0 or k <= 0:
        raise ValueError("Kernel size 'k' must be a positive odd integer")
    if a <= 0:
        raise ValueError("Attenuation factor 'a' should be greater than 0")

    # Create an empty tensor for the kernel
    kernel = torch.zeros((k, k), dtype=torch.float32, device=device)

    # Calculate the center index
    center = k // 2

    # Populate the kernel
    for i in range(k):
        for j in range(k):
            # Calculate the Manhattan distance from the center
            distance = abs(center - i) + abs(center - j)
            # Each layer has a sum value of 1/(a^distance)
            # Using the smart count of pixels per layer
            num_pixels_in_layer = 1 if distance == 0 else 4 * distance
            kernel[i, j] = 1 / (a ** distance * num_pixels_in_layer)
            if distance != 0:
                kernel[i, j] *= (1/2) * scaling_factor

    return kernel

if __name__ == '__main__':
    s = 5  # Size of the kernel
    a = 2  # Attenuation factor
    custom_kernel = generate_custom_kernel(s, a)

    # Display the kernel
    print("Custom Kernel:")
    print(custom_kernel)