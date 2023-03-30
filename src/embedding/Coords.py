import imp
import torch
from torch import nn
 

class Coords(nn.Module):
    """ Activation function to generate values in a symmetric range.
    """
    
    def __init__(self, max_value):
        """
        Args:
            max_value ( numeric ): numeric value to set the symmetric range
        """
        super().__init__()
        self.max_value=max_value

    def forward(self, input):
        """ Apply the activation function.

        Args:
            input ( torch.Tensor ): input tensor

        Returns:
             torch.Tensor : output tensor
        """
        coords=self.max_value*torch.tanh(input)
        return coords