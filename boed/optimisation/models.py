# GPyTorch / BoTorch Stuff
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel


class ExactGP_RBF(ExactGP, GPyTorchModel):
    
    _num_outputs = 1 # to inform GPyTorchModel API

    def __init__(
        self, train_x, train_y, likelihood, 
        lengthscale_prior=None, outputscale_prior=None):
        
        super(ExactGP_RBF, self).__init__(
            train_x, train_y.squeeze(-1), likelihood)
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            RBFKernel(
            ard_num_dims=train_x.shape[-1], 
            lengthscale_prior=lengthscale_prior),
            outputscale_prior=outputscale_prior)

    def forward(self, x):
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return MultivariateNormal(mean_x, covar_x)
    

class ExactGP_Matern(ExactGP, GPyTorchModel):
    
    _num_outputs = 1 # to inform GPyTorchModel API

    def __init__(
        self, train_x, train_y, likelihood, 
        lengthscale_prior=None, outputscale_prior=None):
        
        super(ExactGP_Matern, self).__init__(
            train_x, train_y.squeeze(-1), likelihood)
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(
            nu=2.5, ard_num_dims=train_x.shape[-1], 
            lengthscale_prior=lengthscale_prior),
            outputscale_prior=outputscale_prior)

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return MultivariateNormal(mean_x, covar_x)