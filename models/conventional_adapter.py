from torch import nn


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.hidden_size
        reduction_factor = config.reduction_factor
        self.down_sample_size = self.input_dim // reduction_factor
        self.activation = nn.GELU()
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)

        # self.track_z = config.track_z

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        # if self.track_z:
        #     self.z = z
        output = self.up_sampler(z)
        # Residual connection.
        output = output + x
        return output
