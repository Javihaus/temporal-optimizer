import torch.nn as nn


class HamiltonianNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[512, 256],
        activation="leaky_relu",
        dropout_rate=0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], 2))  # Binary classification

        # Activation function
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.layers[:-1]:  # All layers except the last
            x = self.dropout(self.activation(layer(x)))
        return self.layers[-1](x)  # Last layer (no activation for output)
