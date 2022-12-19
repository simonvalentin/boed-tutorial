import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralSummStats(nn.Module):
    
    """
    Fully-connected neural network written as a child class of torch.nn.Module,
    used to learn the summary statistics of a variable y.

    Attributes
    ----------
    self.fc_var1: torch.nn.Linear object
        Input layer for the first random variable.
    self.fc_var2: torch.nn.Linear object
        Input layer for the second random variable.
    self.layers: torch.nn.ModuleList object
        Object that contains all layers of the neural network.

    Methods
    -------
    forward:
        Forward pass through the fully-connected neural network.
    """

    def __init__(self, var_dim, L=1, H=10, Output_dim=2):
        """
        Parameters
        ----------
        var_dim: int
            Dimensions of the random variable.
        L: int
            Number of hidden layers of the neural network.
            (default is 1)
        H: int or np.ndarray
            Number of hidden units for each hidden layer. If 'H' is an int, all
            layers will have the same size. 'H' can also be an nd.ndarray,
            specifying the sizes of each hidden layer.
            (default is 10)
        """

        super(NeuralSummStats, self).__init__()

        # check for the correct dimensions
        if isinstance(H, (list, np.ndarray)):
            assert len(H) == L, "Incorrect dimensions of hidden units."
            H = list(map(int, list(H)))
        else:
            H = [int(H) for _ in range(L)]

        # Define layers over your two random variables
        self.fc_var = nn.Linear(var_dim, H[0])

        # Define any further layers
        self.layers = nn.ModuleList()
        if L == 1:
            fc = nn.Linear(H[0], Output_dim)
            self.layers.append(fc)
        elif L > 1:
            for idx in range(1, L):
                fc = nn.Linear(H[idx - 1], H[idx])
                self.layers.append(fc)
            fc = nn.Linear(H[-1], Output_dim)
            self.layers.append(fc)
        else:
            raise ValueError('Incorrect value for number of layers.')

    def forward(self, var):
        """
        Forward pass through the neural network.

        Parameters
        ----------
        var1: torch.autograd.Variable
            First random variable.
        var2: torch.autograd.Variable
            Second random variable.
        """

        # Initial layer over random variables
        hidden = F.relu(self.fc_var(var))

        # All subsequent layers
        for idx in range(len(self.layers) - 1):
            hidden = F.relu(self.layers[idx](hidden))

        # Output layer
        output = self.layers[-1](hidden)

        return output
    

class CAT_NSS(nn.Module):
    
    """
    Fully-connected neural network written as a child class of torch.nn.Module,
    used to learn the summary statistics of a variable y.

    Attributes
    ----------
    self.fc_var1: torch.nn.Linear object
        Input layer for the first random variable.
    self.fc_var2: torch.nn.Linear object
        Input layer for the second random variable.
    self.layers: torch.nn.ModuleList object
        Object that contains all layers of the neural network.

    Methods
    -------
    forward:
        Forward pass through the fully-connected neural network.
    """

    def __init__(self, var_dim, L=1, H=10, Output_dim=2, num_measurements=1):
        """
        Parameters
        ----------
        var_dim: int
            Dimensions of the random variable.
        L: int
            Number of hidden layers of the neural network.
            (default is 1)
        H: int or np.ndarray
            Number of hidden units for each hidden layer. If 'H' is an int, all
            layers will have the same size. 'H' can also be an nd.ndarray,
            specifying the sizes of each hidden layer.
            (default is 10)
        """

        super(CAT_NSS, self).__init__()
        
        self.Output_dim = Output_dim
        self.num_measurements = num_measurements
        # self.batch_size = batch_size

        # check for the correct dimensions
        if isinstance(H, (list, np.ndarray)):
            assert len(H) == L, "Incorrect dimensions of hidden units."
            H = list(map(int, list(H)))
        else:
            H = [int(H) for _ in range(L)]
            
        self.networks = nn.ModuleList()
        for i in range(num_measurements):
            S = NeuralSummStats(var_dim=var_dim, L=L, H=H, Output_dim=Output_dim)
            self.networks.append(S)
            
        #self.empty_tensor = torch.empty(
        #    (self.batch_size, self.num_measurements * self.Output_dim))

    def forward(self, var):
        """
        Forward pass through the neural network.

        Parameters
        ----------
        var1: torch.autograd.Variable
            First random variable.
        var2: torch.autograd.Variable
            Second random variable.
        """
        
        # self.empty_tensor.zero_()
        self.empty_tensor = torch.empty(
            (var.shape[0], self.num_measurements * self.Output_dim), device=var.device)

        for i in range(len(self.networks)):
            S_i = self.networks[i](var[:, i, :])
            self.empty_tensor[:, i * self.Output_dim : (i+1) * self.Output_dim] = S_i

        return self.empty_tensor