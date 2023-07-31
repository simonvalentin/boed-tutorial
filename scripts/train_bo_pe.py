import argparse
import numpy
import os
import pandas
import random
import sys
import torch
from tqdm import tqdm as tqdm

from boed.simulators.bandits import BanditDatasetPE_Multiple, sim_bandit_prior
from boed.optimisation.bayesopt import evaluate_summ, train_GP_scipy
from boed.optimisation.models import ExactGP_Matern

from gpytorch.likelihoods import GaussianLikelihood

from ax import Data, Experiment, Metric, Runner, SearchSpace, Objective, OptimizationConfig, optimize
from ax.modelbridge import get_sobol
from ax.modelbridge.factory import get_botorch
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.storage.json_store.encoder import object_to_json
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.service.ax_client import AxClient
from ax.service.utils.best_point import get_best_raw_objective_point
from ax.utils.common.result import Ok

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ====================================
# Hyper-parameters and file arguments
# ====================================


parser = argparse.ArgumentParser(description='Bandit BO Model Parser')

# SIMULATOR PARAMS

# Number of Bandit Arms
parser.add_argument('--arms', type=int, default=3, metavar='D',
                    help='Number of arms (default: 3)')
# Number of Trials per Block
parser.add_argument('--trials', type=int, default=30, metavar='T',
                    help='Number of trials per block (default: 30)')
# Number of Blocks
parser.add_argument('--blocks', type=int, default=2, metavar='T',
                    help='Number of blocks (default: 2)')
# Prior type ('empirical' or 'uninformed')
parser.add_argument('--priortype', type=str, default='uninformed', metavar='PT',
                    help='Type of prior distribution (default: uninformed)')
# Prior type ('empirical' or 'uninformed')
parser.add_argument('--simmodel', type=int, default=0, metavar='SM',
                    help='Model indicator: 0, 1 or 2 (default: 0)')

# NN PARAMS

# Datasize for simulations
parser.add_argument('--datasize', type=int, default=10000, metavar='DS',
                    help='Input data size for training (default: 10000)')
# Batch-size for training
parser.add_argument('--batchsize', type=int, default=0, metavar='BS',
                    help='Input batch size for training (default: 0)')
# Number of layers of NN
parser.add_argument('--layers', type=int, default=2, metavar='L',
                    help='Number of layers of NN (default: 2)')
# Number of units per layer
parser.add_argument('--units', nargs='+', type=int, default=50, metavar='H',
                    help='Hidden units of NN (default: 50)')
# Number of epochs for training
parser.add_argument('--epochs', type=int, default=1000, metavar='E',
                    help='Number of epochs to train (default: 10000)')
# Learning rate for NN parameters
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='NN learning rate (default: 1e-3)')
# Weight decay for NN parameters
parser.add_argument('--wd', type=float, default=1e-3, metavar='WD',
                    help='NN weight decay (default: 1e-3)')
# Prior type ('empirical' or 'uninformed')
parser.add_argument('--scheduler', type=str, default='none', metavar='SC',
                    help='What type of scheduler to use (default: none)')
# Prior type ('empirical' or 'uninformed')
parser.add_argument('--plateau-factor', type=float, default=0.5, metavar='PF',
                    help='Factor of Plateau scheduler (default: 0.5)')
# Prior type ('empirical' or 'uninformed')
parser.add_argument('--plateau-patience', type=float, default=25, metavar='PP',
                    help='Patience of Plateau scheduler (default: 25)')

# SUMM STATS PARAMS

# Number of layers of SummStat NN
parser.add_argument('--summ-layers', type=int, default=2, metavar='SL',
                    help='Number of layers of NN (default: 2)')
# Number of units per layer for SummStat NN
parser.add_argument('--summ-units', nargs='+', type=int, default=50, metavar='SU',
                    help='Hidden units of NN (default: 50)')
# Number of dimensions of SummStat output per measurement
parser.add_argument('--summ-output', type=int, default=5, metavar='SO',
                    help='Summary Stats output dimensions (default: 5)')

# BO PARAMS

# Number of initial trials (SOBOL)
parser.add_argument('--inits', type=int, default=5, metavar='IN',
                    help='Number of BO initialisation (default: 5)')
# Number of BO trials
parser.add_argument('--evals', type=int, default=20, metavar='BO',
                    help='Number of BO evaluations (default: 20)')

# MISC

# Random seed for reproducibility
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
# Job ID for array jobs
parser.add_argument('--job-id', type=int, default=-1, metavar='ID',
                    help='Job ID for array jobs (default: -1)')
# Job ID for array jobs
parser.add_argument('--filename', type=str, default='bandit_bo', metavar='FN',
                    help='File name for saving (default: bandit_bo)')

args = parser.parse_args()

# Set to None if you don't want to use batches at all
if args.batchsize == 0:
    args.batchsize = args.datasize

# If no seed is provided, a random number is drawn
if args.seed == 0:
    args.seed = torch.randint(0, 2**32, (1, )).item()
    seed_torch(args.seed)
else:
    seed_torch(args.seed)
    
# if multiple layers are required but list of hidden units has only one entry
if len(args.summ_units) == 1 and args.summ_layers > 1:
    args.summ_units = args.summ_units[0]


# =========================================
# Prior samples and model hyper-parameters
# =========================================


# Get regular prior samples
prior = sim_bandit_prior(args.datasize, prior=args.priortype, simmodel=args.simmodel)

# put relevant hyper-parameters into a dict
modelparams = {
    'batchsize': args.batchsize,
    'layers': args.layers,
    'hidden': args.units,
    'lr': args.lr,
    'weight_decay': args.wd,
    'num_epochs': args.epochs,
    'num_workers': 0,
    'num_measurements': args.blocks,
    'summary_stats': True,
    'summ_L': args.summ_layers,
    'summ_H': args.summ_units,
    'summ_out': args.summ_output,
    'scheduler': args.scheduler,
    'plateau_factor': args.plateau_factor,
    'plateau_patience': args.plateau_patience
}


# =================
# Helper Functions
# =================


# need to wrapt the evaluater function for BO
def objective_designs(parameters):
    
    # extract designs
    d = numpy.array([parameters.get("d{}".format(i)) for i in range(1, len(parameters) + 1)]).reshape(-1, 1)

    # get validation score
    val, _, _, _, _ = evaluate_summ(
        d, prior, device, modelparams, BanditDatasetPE_Multiple, 
        simbar=False, bar=False, num_trials=args.trials, num_blocks=args.blocks,
        num_arms=args.arms, valid=True, simmodel=args.simmodel)

    return val


# Set Up the GP Model
def get_and_fit_model(Xs, Ys, kernel='Matern', optmethod='scipy', **kwargs):
    
    train_x, train_y = Xs[0], Ys[0]

    # Create GP Model
    if kernel=='RBF':  # rbf kernel
        likelihood = GaussianLikelihood(noise_prior=None)
        model = ExactGP_RBF(
            train_x, train_y, likelihood, 
            lengthscale_prior=None, outputscale_prior=None)
    elif kernel=='Matern':  # matern kernel
        likelihood = GaussianLikelihood(noise_prior=None)
        model = ExactGP_Matern(
            train_x, train_y, likelihood, 
            lengthscale_prior=None, outputscale_prior=None)
    else:
        raise NotImplementedError('This Kernel is not yet supported')
        
    # Fit GP Model
    if optmethod=='torch':  # Use a torch Optimiser
        model, _, _ = train_GP_torch(
            model, likelihood, train_x, train_y, 
            lr=0.001, train_iter=1000, trainbar=False)
    elif optmethod=='scipy':
        # TRAIN RBF GP
        model, _ = train_GP_scipy(
            model, likelihood, train_x, train_y, 
            lr=0.001, train_iter=1000)
        
    return model


# mock runner for Ax-platform
class MockRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}
    

# custom metric using lower bound evaluations
class LowerBoundMetric(Metric):
    def fetch_trial_data(self, trial):  
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": objective_designs(params),
                "sem": 0.0,
                "trial_index": trial.index,
            })
        return Ok(value=Data(df=pandas.DataFrame.from_records(records)))


# ============================
# Bayesian optimisation setup
# ============================


# total number of designs
num_designs = args.arms * args.blocks

# Create the Design Search Space
parameters = [{"name": "d{}".format(i), 
            "type": "range", "bounds": [0.0, 1.0], 
            "value_type": "float"} for i in range(1, num_designs+1)]
exp_parameters = [AxClient.parameter_from_json(p) for p in parameters]
space = SearchSpace(parameters=exp_parameters, parameter_constraints=None)

# Instantiate an OptimizationConfig object
param_names = [p.name for p in exp_parameters]
optimization_config = OptimizationConfig(
    objective = Objective(
        metric=LowerBoundMetric(name='lower_bound', lower_is_better=False), 
        minimize=False,
    ),
)

# Create the Experiment
exp = Experiment(
    name=f"Bandit_BO_{args.arms}arms",
    search_space=space,
    optimization_config=optimization_config,
    runner=MockRunner()
)


# ============================
# Run Bayesian optimisation
# ============================


# Initial BO Evaluations
print(f"Starting {args.inits} Initial Sobol Evaluations")
sobol = get_sobol(exp.search_space)
for i in tqdm(range(args.inits)):

    # create initial designs sampled from a sobol sequence
    trial = exp.new_trial(generator_run=sobol.gen(1))
    
    # evaluate the objective function at the initial designs
    trial.run()
    trial.mark_completed()

# Sequential BO Evaluations
print(f"Starting {args.evals - args.inits} BO Evaluations")
for i in tqdm(range(args.evals - args.inits)):
    
    # re-instantiate BOTORCH model and create the batch
    model = get_botorch(
        experiment=exp,
        data=exp.fetch_data(),
        search_space=exp.search_space,
        model_constructor=get_and_fit_model
    )
    batch = exp.new_trial(generator_run=model.gen(1))
    
    # run the BO procedure
    batch.run()
    batch.mark_completed()
    
# Run it one more time to get the last GP fit
model = get_botorch(
        experiment=exp,
        data=exp.fetch_data(),
        search_space=exp.search_space,
        model_constructor=get_and_fit_model
)
batch = exp.new_trial(generator_run=model.gen(1))
batch.run()
batch.mark_completed()
print("Done!")


# ===============
# Saving results
# ===============


# update Ax-Platform registries with our custom metric and mock runner
_, encoder_reg, _ = register_metric(LowerBoundMetric)
_, encoder_reg, _ = register_runner(MockRunner, encoder_registry=encoder_reg)

# convert experiment data to json
json_experiment = object_to_json(exp, encoder_registry=encoder_reg)

# get best evaluation
best_parameters, all_values = get_best_raw_objective_point(exp)
values = (
    {k: v[0] for k, v in all_values.items()},  # v[0] is mean
    {k: {k: v[1] * v[1]} for k, v in all_values.items()},  # v[1] is sem
)

# Prepare data to save
bo_dict = {
    'best_parameters': best_parameters,
    'values': values,
    'json_experiment': json_experiment}

# Add arguments to parser
save_dict = dict(bo_dict, **vars(args))

# save torch_data
directory = '../'
torch.save(save_dict, directory + 'data/{}.pt'.format(args.filename))
