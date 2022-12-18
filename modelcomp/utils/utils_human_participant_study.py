from typing import List, Optional, Tuple, Union

# general imports
import copy
import json
import numpy
import random
from scipy import integrate
import scipy.stats as sts
from tqdm import tqdm as tqdm
import torch
device = torch.device('cpu')

# BED imports
from modelcomp.networks.fullyconnected import FullyConnected
from modelcomp.networks.summstats import NeuralSummStats, CAT_NSS
from modelcomp.simulators.bandits import simulate_bandit_batch, sim_bandit_prior


def extract_evals_custom(data):
    
    experiment = object_from_json(data['json_experiment'])
    
    # Get designs and MI evals
    X = list()
    Y = list()
    for t in experiment.trials.values():
        
        if t.index < len(experiment.trials.values())-1:
            
            d = list(t.arm.parameters.values())
            y = t.objective_mean
            
            X.append(d)
            Y.append(y)
    X = numpy.array(X)
    Y = numpy.array(Y)
    
    return X, Y


def transform_data(user, desired_design, num_arms=3):    
    
    # copy user over
    user_dict = copy.deepcopy(user)
    
    # Obtain the desired order of blocks to be swapped
    d1 = user_dict['design']
    d2 = desired_design
    idxs_swap = list()
    for i in range(len(d2)):
        for j in range(len(d1)):
            if sorted(d2[i]) == sorted(d1[j]):
                if j not in idxs_swap:
                    idxs_swap.append(j)

    # swap blocks of designs, choices and rewards
    user_dict['design'] = numpy.array(user_dict['design'])[numpy.array(idxs_swap)].tolist()
    user_dict['choices'] = numpy.array(user_dict['choices'])[numpy.array(idxs_swap)].tolist()
    user_dict['rewards'] = numpy.array(user_dict['rewards'])[numpy.array(idxs_swap)].tolist()

    # note down whether or not we swapped blocks
    if idxs_swap[::-1] == list(range(len(user_dict['design']))):
        user_dict['swapped_blocks'] = True
    else:
        user_dict['swapped_blocks'] = False
        
    # Obtain the desired order of arms within each block
    idxs_swap_arms = list()
    for k in range(len(user_dict['design'])):
    
        idxs_tmp = list()
        for i in range(len(user_dict['design'][k])):
            for j in range(len(desired_design[k])):
                # print(i, j)
                if user_dict['design'][k][i] == desired_design[k][j]:
                    if j not in idxs_tmp:
                        idxs_tmp.append(j)
                        break

        # reassign. // slicing produced wrong results
        arms_new = [0 for _ in range(len(user_dict['design'][k]))]
        for a in range(num_arms):
            arms_new[idxs_tmp[a]] = user_dict['design'][k][a]
        user_dict['design'][k] = arms_new
        idxs_swap_arms.append(idxs_tmp)

    # Swap order of arms within each block of choices
    choices_tmp = numpy.array(copy.deepcopy(user_dict['choices']))
    for k in range(len(user_dict['design'])):
    
        list_actions = list()
        for i in range(len(user_dict['design'][k])):
            idxs_action = numpy.argwhere(
                numpy.array(user_dict['choices'][k]) == i).reshape(-1)
            list_actions.append(idxs_action)

        for i in range(len(user_dict['design'][k])):
            if len(list_actions[i]) == 0:
                continue
            elif len(list_actions[i]) == 1:
                choices_tmp[k][list_actions[i][0]] = idxs_swap_arms[k][i]
            else:
                choices_tmp[k][list_actions[i]] = idxs_swap_arms[k][i]
    user_dict['choices'] = choices_tmp.tolist()

    return user_dict


def combine_choices_rewards(user):
    
    choices = user['choices']
    rewards = user['rewards']
    
    y_obs = torch.tensor([choices[i] + rewards[i] for i in range(len(choices))])
    
    return y_obs.float()


def get_md_data_only(user, md_blocks=2):
    
    user_dict = copy.deepcopy(user)
    
    user_dict['design'] = user_dict['design'][:md_blocks]
    user_dict['choices'] = user_dict['choices'][:md_blocks]
    user_dict['rewards'] = user_dict['rewards'][:md_blocks]
    
    return user_dict


def get_pe_data_only(user, md_blocks=2):
    
    user_dict = copy.deepcopy(user)
    
    user_dict['design'] = user_dict['design'][md_blocks:]
    user_dict['choices'] = user_dict['choices'][md_blocks:]
    user_dict['rewards'] = user_dict['rewards'][md_blocks:]
    
    return user_dict


def get_trained_models(modelparams, filepath_FC, filepath_Summ, dim1=1):
    
    L = modelparams['layers']
    H = modelparams['hidden']
    summ_L = modelparams['summ_L']
    summ_H = modelparams['summ_H']
    summ_out = modelparams['summ_out']
    num_measurements = modelparams['num_measurements']

    # dim1 = 1
    dim2 = 60

    model_tr = FullyConnected(
        var1_dim=dim1, var2_dim=summ_out * num_measurements, L=L, H=H)
    model_summ_tr = CAT_NSS(
        var_dim=dim2, L=summ_L, H=summ_H, Output_dim=summ_out, 
        num_measurements=num_measurements)

    # Load in Weights
    model_tr.load_state_dict(torch.load(filepath_FC, map_location=torch.device('cpu')), strict=True)
    model_tr.eval()
    model_summ_tr.load_state_dict(torch.load(filepath_Summ, map_location=torch.device('cpu')), strict=True)
    model_summ_tr.eval()
    
    return model_tr, model_summ_tr


def get_dataloader(design, prior, batchsize=512, num_workers=0, device=torch.device("cpu")):
    
    # define data set
    dataset = BanditDatasetMD_Multiple(
        design, prior, device, num_trials=30, num_blocks=2, num_arms=3, n_cores=1, simbar=True)

    dl = torch.utils.data.DataLoader(
        dataset, batch_size=batchsize, drop_last=True, shuffle=True, num_workers=num_workers)
    
    return dl


def validate_summ(model, model_summ, validloader, device):
    
    scores = list()
    for x, y in validloader:
        
        x, y = x.to(device), y.to(device)
        
        # compute summary statistics
        Sy = model_summ(y)
        
        # compute loss and store in list
        loss = minef_loss(x, Sy, model, device)
        scores.append(loss.cpu().data.numpy())
        
    return numpy.mean(-numpy.array(scores))


def weighted_average(posts, weights):
    
    ps_norm = posts / numpy.sum(posts, axis=1).reshape(-1, 1)
    ws_norm = weights / numpy.sum(weights)
    
    ps_norm *= weights.reshape(-1, 1)
    
    post_mean = numpy.mean(ps_norm, axis=0) / numpy.sum(numpy.mean(ps_norm, axis=0))
    
    return post_mean


def get_pe_posterior_histograms_optimal(
    users: list,
    model_list: list,
    model_summ_list: list,
    prior_samples: numpy.ndarray,
    hist_bins_list: List[numpy.ndarray],
    simmodel: int = 0,
    num_resample: int = 10_000,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    
    posts_ensemble = list()
    corrs_ensemble = list()
    for user in tqdm(users):
        
        # real-world observation
        y_obs = combine_choices_rewards(user).unsqueeze(0)

        posts_single_network = list()
        corrs_single_network = list()
        for i in tqdm(range(len(model_list)), disable=True):

            # extract the correct models
            model_tr = model_list[i]
            model_summ_tr = model_summ_list[i]

            # get summary statistics for the observation
            Sy_obs = model_summ_tr(y_obs)

            # select parameters according to the correct model
            if simmodel == 0:
                prior_select = prior_samples[:,1:4]
            elif simmodel == 1:
                prior_select = prior_samples[:,4:6]
            elif simmodel == 2:
                prior_select = prior_samples[:,6:-1]

            # convert to tensors
            X = torch.tensor(prior_select, dtype=torch.float, device=device)
            X.to(X)
            Y = torch.cat(len(X)*[Sy_obs])
            Y.to(device);

            # Compute importance weights
            T = model_tr(X, Y.float()).data.numpy().reshape(-1)
            post_weights = numpy.exp(T - 1)
            ws_norm = post_weights / numpy.sum(post_weights)

            # get posterior samples
            idx_samples = random.choices(range(len(ws_norm)), weights=ws_norm, k=num_resample)
            post_samples = prior_select[idx_samples]

            # compute correlations
            post_corr = numpy.corrcoef(post_samples.T)
            corrs_single_network.append(post_corr)

            # get value of histograms
            hist_tmp = list()
            for idx in range(post_samples.shape[-1]):
                p = post_samples[:, idx]
                hist, _ = numpy.histogram(p, bins=hist_bins_list[idx], density=True)
                hist_tmp.append(hist.tolist())
            posts_single_network.append(hist_tmp)
        posts_ensemble.append(posts_single_network)
        corrs_ensemble.append(corrs_single_network)
    
    return numpy.array(posts_ensemble), numpy.array(corrs_ensemble)


def get_pe_posterior_histograms_baseline(
    users: list,
    model_dict: dict,
    model_summ_dict: dict,
    prior_samples: numpy.ndarray,
    hist_bins_list: List[numpy.ndarray],
    simmodel: int = 0,
    num_resample: int = 10_000,
    baseline_allocation: Optional[numpy.ndarray] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    
    posts_ensemble = list()
    corrs_ensemble = list()
    for i in tqdm(range(len(users))):
        
        if baseline_allocation[i] == simmodel:  # only select the appropriate models

            posts_single_network = list()
            corrs_single_network = list()
            condition = users[i]['conditionS1']
            for j in range(len(model_dict[condition])):

                # get the appropriate models
                model_tr = model_dict[condition][j]
                model_summ_tr = model_summ_dict[condition][j]

                # real-world observation
                # TODO: Check if this should be outside this inner loop
                y_obs = combine_choices_rewards(users[i]).unsqueeze(0)

                # compute summary statistics of observation
                Sy_obs = model_summ_tr(y_obs)

                # select parameters according to the correct model
                if simmodel == 0:
                    prior_select = prior_samples[:,1:4]
                elif simmodel == 1:
                    prior_select = prior_samples[:,4:6]
                elif simmodel == 2:
                    prior_select = prior_samples[:,6:-1]

                # convert to tensors
                X = torch.tensor(prior_select, dtype=torch.float, device=device)
                X.to(X)
                Y = torch.cat(len(X)*[Sy_obs])
                Y.to(device);

                # Compute importance weights
                T = model_tr(X, Y.float()).data.numpy().reshape(-1)
                post_weights = numpy.exp(T - 1)
                ws_norm = post_weights / numpy.sum(post_weights)

                # get posterior samples
                idx_samples = random.choices(range(len(ws_norm)), weights=ws_norm, k=num_resample)
                post_samples = prior_select[idx_samples]

                # compute correlations
                post_corr = numpy.corrcoef(post_samples.T)
                corrs_single_network.append(post_corr)

                # get value of histograms
                hist_tmp = list()
                for idx in range(post_samples.shape[-1]):
                    p = post_samples[:, idx]
                    hist, _ = numpy.histogram(p, bins=hist_bins_list[idx], density=True)
                    hist_tmp.append(hist.tolist())
                posts_single_network.append(hist_tmp)
            posts_ensemble.append(posts_single_network)
            corrs_ensemble.append(corrs_single_network)
        else:
            continue

    return numpy.array(posts_ensemble), numpy.array(corrs_ensemble)


def get_norm_constant(joint_densities, spaces):
    
    N_GRIDS = [len(sp) for sp in spaces]
    
    joint_resh = joint_densities.reshape(*N_GRIDS)
        
    Z = joint_resh
    for i, sp in enumerate(spaces[::-1]):
        Z = integrate.simps(Z, sp, axis=len(spaces)-(i+1))
    
    return Z


def get_differential_entropy(joint_densities, spaces):
    
    N_GRIDS = [len(sp) for sp in spaces]
    
    integ = joint_densities * numpy.log(joint_densities + 1e-32)
    integ[numpy.isnan(integ)] = 0
    integ[numpy.isinf(integ)] = 0
    
    integ_resh = integ.reshape(*N_GRIDS)
    
    Z = integ_resh
    for i, sp in enumerate(spaces[::-1]):
        Z = integrate.simps(Z, sp, axis=len(spaces)-(i+1))
        
    return -1 * Z


def get_pe_entropies_optimal(
    users: list,
    model_list: list,
    model_summ_list: list,
    prior_samples: numpy.ndarray,
    grid_list: List[numpy.ndarray],
    simmodel: int = 0,
    num_resample: int = 10_000,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    
    # get a meshgrid of the parameter space
    th_meshgrid = numpy.meshgrid(*grid_list)
    
    # get a list of meshgrid positions
    positions = numpy.vstack(tuple(th.ravel() for th in th_meshgrid)).T
    
    entropy_avg = list()
    entropy_ind = list()
    for user in tqdm(users):

        # real-world observation
        y_obs = combine_choices_rewards(user).unsqueeze(0)

        posts_single_network = list()
        entropy_single_network = list()
        for i in tqdm(range(len(model_list)), disable=True):

            # get the appropriate models
            model_tr = model_list[i]
            model_summ_tr = model_summ_list[i]

            # compute summary statistics of observation
            Sy_obs = model_summ_tr(y_obs)

            # select parameters according to the correct model
            if simmodel == 0:
                prior_select = prior_samples[:,1:4]
            elif simmodel == 1:
                prior_select = prior_samples[:,4:6]
            elif simmodel == 2:
                prior_select = prior_samples[:,6:-1]

            # convert to tensors
            X = torch.tensor(prior_select, dtype=torch.float, device=device)
            X.to(X)
            Y = torch.cat(len(X)*[Sy_obs])
            Y.to(device);

            # Compute importance weights
            T = model_tr(X, Y.float()).data.numpy().reshape(-1)
            post_weights = numpy.exp(T - 1)
            ws_norm = post_weights / numpy.sum(post_weights)

            # get posterior samples
            idx_samples = random.choices(range(len(ws_norm)), weights=ws_norm, k=num_resample)
            post_samples = prior_select[idx_samples]

            # create kde of post samples
            kernel = sts.gaussian_kde(post_samples.T)
            post_pdf = kernel(positions.T)
            post_pdf = post_pdf.reshape(th_meshgrid[0].shape)
            posts_single_network.append(post_pdf)

            # normalize joint density
            Z = get_norm_constant(
                post_pdf.reshape(-1),
                spaces = tuple(grid_list))
            post_pdf = post_pdf / Z

            # compute entropy
            dH = get_differential_entropy(
                post_pdf.reshape(-1),
                spaces = tuple(grid_list))
            entropy_single_network.append(dH)
        entropy_ind.append(entropy_single_network)

        # compute mean of KDEs
        pp = numpy.mean(numpy.array(posts_single_network), axis=0)

        # normalize joint density
        Z = get_norm_constant(
            pp.reshape(-1),
            spaces = tuple(grid_list))
        pp = pp / Z

        # compute entropy
        dH = get_differential_entropy(
            pp.reshape(-1),
            spaces = tuple(grid_list))

        entropy_avg.append(dH)
        
    return numpy.array(entropy_avg), numpy.array(entropy_ind)


def get_pe_entropies_baseline(
    users: list,
    model_dict: dict,
    model_summ_dict: dict,
    prior_samples: numpy.ndarray,
    grid_list: List[numpy.ndarray],
    simmodel: int = 0,
    num_resample: int = 10_000,
    baseline_allocation: Optional[numpy.ndarray] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    
    # get a meshgrid of the parameter space
    th_meshgrid = numpy.meshgrid(*grid_list)
    
    # get a list of meshgrid positions
    positions = numpy.vstack(tuple(th.ravel() for th in th_meshgrid)).T

    entropy_avg = list()
    entropy_ind = list()
    for i in tqdm(range(len(users))):

        if baseline_allocation[i] == simmodel:

            posts_single_network = list()
            entropy_single_network = list()
            condition = users[i]['conditionS1']
            for j in range(len(model_dict[condition])):

                # get the appropriate models
                model_tr = model_dict[condition][j]
                model_summ_tr = model_summ_dict[condition][j]

                # real-world observation
                y_obs = combine_choices_rewards(users[i]).unsqueeze(0)

                # compute summary statistics of observation
                Sy_obs = model_summ_tr(y_obs)

                # select parameters according to the correct model
                if simmodel == 0:
                    prior_select = prior_samples[:,1:4]
                elif simmodel == 1:
                    prior_select = prior_samples[:,4:6]
                elif simmodel == 2:
                    prior_select = prior_samples[:,6:-1]

                # convert to tensors
                X = torch.tensor(prior_select, dtype=torch.float, device=device)
                X.to(X)
                Y = torch.cat(len(X)*[Sy_obs])
                Y.to(device);

                # Compute importance weights
                T = model_tr(X, Y.float()).data.numpy().reshape(-1)
                post_weights = numpy.exp(T - 1)
                ws_norm = post_weights / numpy.sum(post_weights)

                # get posterior samples
                idx_samples = random.choices(range(len(ws_norm)), weights=ws_norm, k=num_resample)
                post_samples = prior_select[idx_samples]

                # create kde of post samples
                kernel = sts.gaussian_kde(post_samples.T)
                post_pdf = kernel(positions.T)
                post_pdf = post_pdf.reshape(th_meshgrid[0].shape)
                posts_single_network.append(post_pdf)

                # normalize joint density
                Z = get_norm_constant(
                    post_pdf.reshape(-1),
                    spaces = tuple(grid_list))
                post_pdf = post_pdf / Z
                # compute entropy
                dH = get_differential_entropy(
                    post_pdf.reshape(-1),
                    spaces = tuple(grid_list))
                entropy_single_network.append(dH)
            entropy_ind.append(entropy_single_network)

            # compute mean of KDEs
            pp = numpy.mean(numpy.array(posts_single_network), axis=0)

            # normalize joint density
            Z = get_norm_constant(
                pp.reshape(-1),
                spaces = tuple(grid_list))
            pp = pp / Z

            # compute entropy
            dH = get_differential_entropy(
                pp.reshape(-1),
                spaces = tuple(grid_list))

            entropy_avg.append(dH)
        else:
            continue
        
    return numpy.array(entropy_avg), numpy.array(entropy_ind)



def plot_correlations(
    corrs, fig, ax, num_decimals=3, colorbar=True,
    numsize=20, cmap='Blues', size=3):
    
    plot = corrs

    # Limits for the extent
    x_start = 0
    x_end = size
    y_start = 0
    y_end = size

    extent = [x_start, x_end, y_start, y_end]

    plot = plot[::-1]
    im = ax.imshow(
        plot, origin='lower', extent=extent, 
        interpolation=None, cmap=cmap, vmin=-1, vmax=1)

    # Add the text
    jump_x = (x_end - x_start) / (2.0 * size)
    jump_y = (y_end - y_start) / (2.0 * size)
    x_positions = numpy.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
    y_positions = numpy.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = plot[y_index, x_index]
            text_x = x + jump_x
            text_y = y + jump_y
            if label < 0.6:
                color='black'
            else:
                color='white'
            
            if label > 0:
                label = str(label)[:2+num_decimals]
            elif label < 0:
                label = str(label)[:3+num_decimals]
            else:
                raise ValueError()
            ax.text(text_x, text_y, label, color=color, ha='center', va='center', size=numsize)

    if colorbar:
        fig.colorbar(im, fraction=0.046, pad=0.04)
    
    return fig, ax, im