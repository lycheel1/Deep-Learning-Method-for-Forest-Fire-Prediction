import torch
import torch.nn as nn


class Embedding_block(nn.Module):
    def __init__(self, embedding_dim, unique_values=None):
        super(Embedding_block, self).__init__()

        # default values set for fuel map
        if unique_values is None:
            unique_values = [  0.,   1.,   2.,   3.,   4.,   5.,   7.,  11.,  12.,  13.,  31., 101., 102., 106.,
                            425., 635., 650., 665.]

        self.unique_values = torch.Tensor(unique_values).cuda()  # Initial tensor creation

        self.embedding_dim = embedding_dim

        # Initialize the embedding layer
        self.embedding = nn.Embedding(num_embeddings=len(unique_values), embedding_dim=embedding_dim)
        self.BN = nn.BatchNorm2d(embedding_dim)

    def forward(self, x_2d_in):

        # Extract the channel to transform
        fuel_ch = x_2d_in[:, 3, :, :]

        # Create mask and map to indices
        mask = fuel_ch.unsqueeze(-1) == self.unique_values
        matching_indices = torch.argmax(mask.int(), dim=-1)

        # Apply embedding and reshape
        embedded_fuel = self.embedding(matching_indices)
        embedded_reshaped_fuel = embedded_fuel.permute(0, 3, 1, 2)
        embedded_reshaped_fuel = self.BN(embedded_reshaped_fuel)

        # Concatenate the transformed channel back with the input tensor
        x_2d_out = torch.cat((x_2d_in[:, :3, :, :], embedded_reshaped_fuel, x_2d_in[:, 4:, :, :]), dim=1)

        return x_2d_out


