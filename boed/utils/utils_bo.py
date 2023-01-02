from typing import Dict, Optional

import json
import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import torch

device = torch.device('cpu')


def maximise_gp_mean(
    model,
    arms: int = 2,
    blocks: int = 1,
    num_restarts: int =10,
    lr: float =1e-1,
    max_iter: int = 1000,
    trainbar: bool = False,
    verbose: bool = False
):
    
    optima = list()
    GP_max = list()
    
    for _ in tqdm(range(num_restarts)):
    
        # initialise design
        init = numpy.random.uniform(0, 1, size=(1, arms * blocks))
        d = torch.tensor(init, dtype=torch.float, requires_grad=True)
            
        # Use the adam optimizer
        optimizer = torch.optim.Adam([d], lr=lr, amsgrad=True)
    
        for i in tqdm(range(max_iter), disable=not trainbar):
            
            optimizer.zero_grad()
            out = model(d.double()).loc
            loss = -out
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                d = d.clamp_(min=0, max=1)
                
            if verbose and (i+1) % 100 == 0:
                print('Iters {}/{} with GP output: {}'.format(i+1, max_iter, out.detach().data.numpy()[0]))
            
        optima.append(d.detach().tolist()[0])
        GP_max.append(out.detach().tolist()[0])
    optima = numpy.array(optima)
    GP_max = numpy.array(GP_max)
    
    idxs = numpy.argsort(GP_max)[::-1]  # sort according to highest GP mean value
        
    return optima[idxs][0], optima[idxs], GP_max[idxs]


def plot_slices(
    model,
    slice_dict: Dict[str, Optional[float]],
    density: int = 50,
    std: int = True
):
    
    """
    slice_dict should be of the shape {'d1': None, 'd2': None, 'd3': 0} 
    if we want to take a slice at d3=0.
    """
    
    # Prepare Grid
    density = 50
    xx = torch.linspace(0, 1, density)
    test_1, test_2 = torch.meshgrid(xx, xx)
    test_x = torch.cat((test_1.reshape(-1, 1), 
                        test_2.reshape(-1, 1)), axis=1)
    x, y = numpy.meshgrid(xx.numpy(), xx.numpy())
    
    # Append slices to Grid
    test_x = list()
    plotting_vars = list()
    plotting_slices = list()
    i = 0
    for key, value in slice_dict.items():
        if value is None:
            # Need to put the meshgrid into the test vector
            idx = int(key[-1]) - 1
            plotting_vars.append(key)
            if i == 0:
                test = test_1.reshape(-1, 1)
                i += 1
            elif i == 1:
                test = test_2.reshape(-1, 1)
            test_x.append(test.tolist())
        elif value is not None:
            # Put the slice vector into the test vector
            idx = int(key[-1]) - 1
            test = torch.ones(len(test_1.reshape(-1, 1))).reshape(-1, 1) * value
            test_x.append(test.tolist())
            plotting_slices.append(key)
    test_x = torch.tensor(test_x, dtype=torch.double).T
    
    # Compute GP Mean and Stddev
    model.eval();
    output = model(test_x)
    mean_test = output.loc.detach().numpy()
    stddev_test = output.stddev.detach().numpy()
    Z = numpy.array(mean_test).reshape(len(xx.numpy()), len(xx.numpy()))
    Z_stddev = numpy.array(stddev_test).reshape(len(xx.numpy()), len(xx.numpy()))
    
    if std:

        fig = plt.figure(figsize=(12, 4))

        ax = fig.add_subplot(121)
        cp = ax.contourf(x, y, Z, levels=20)
        fig.colorbar(cp)
        ax.set_xlabel(plotting_vars[0])
        ax.set_ylabel(plotting_vars[1])
        ax.set_title('GP Mean ({} Arms)'.format(len(slice_dict)))

        ax = fig.add_subplot(122)
        cp = ax.contourf(x, y, Z_stddev, levels=20, cmap=plt.cm.coolwarm)
        fig.colorbar(cp)
        ax.set_xlabel(plotting_vars[0])
        ax.set_ylabel(plotting_vars[1])
        ax.set_title('GP Std. Dev. ({} Arms)'.format(len(slice_dict)))

    else:

        fig = plt.figure(figsize=(6, 4))

        ax = fig.add_subplot(111)
        cp = ax.contourf(x, y, Z, levels=20)
        fig.colorbar(cp)
        ax.set_xlabel('d1')
        ax.set_ylabel('d2')
        ax.set_title('GP Mean ({} Arms)'.format(len(slice_dict)))

    plt.tight_layout();
 
    return fig, ax


def extract_evals_custom(data):
   
    # get json experiment
    data_json = data['json_experiment']
    
    # identify number of trials
    num_trials = len(data_json['trials'])

    # Get designs and MI evals
    X = list()
    Y = list()
    for index in range(num_trials-1):
        
        # extract the designs from the data
        d = list(data_json['trials'][index]['generator_run']['arms'][0]['parameters'].values())

        # extract the objective evaluations from the data
        result_json = data_json['data_by_trial'][index]['value'][0][1]['df']['value']
        y = json.loads(result_json)['mean']['0']

        X.append(d)
        Y.append(y)
    
    return numpy.array(X), numpy.array(Y)
