import torch
from typing import Callable


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.activation = activation
        self.initializer = initializer

        hidden_layers = []
        for i in range(self.hidden_count):
            if i == 0:
                hidden_layers.append(
                    torch.nn.Linear(self.input_size, self.hidden_size)
                )
            else:
                hidden_layers.append(
                    torch.nn.Linear(self.hidden_size, self.hidden_size)
                )
            hidden_layers.append(self.activation())
        
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        self.output_layer = torch.nn.Linear(self.hidden_size, self.num_classes)
        initializer(self.output_layer.weight)


    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        return self.output_layer(self.hidden_layers(x))
