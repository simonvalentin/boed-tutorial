from typing import List, Tuple

# general imports
import numpy
import random
import scipy.stats as sts
from tqdm import tqdm as tqdm
import torch
device = torch.device('cpu')

# BOED imports
from modelcomp.networks.fullyconnected import FullyConnected
from modelcomp.networks.summstats import CAT_NSS
from modelcomp.simulators.bandits import simulate_bandit_batch

# matplotlib imports
import matplotlib.pyplot as plt


def load_trained_models(
    modelparams: dict,
    filepath_FC: str,
    filepath_Summ: str,
    dim1: int = 1,
    dim2: int = 60,
) -> Tuple[FullyConnected, CAT_NSS]:
    """Load trained models and instantiate them."""

    # load the fully-connected main neural network
    model_tr = FullyConnected(
        var1_dim=dim1,
        var2_dim=modelparams['summ_out'] * modelparams['num_measurements'],
        L=modelparams['layers'],
        H=modelparams['hidden'],
    )
    # load the summary statistics models
    model_summ_tr = CAT_NSS(
        var_dim=dim2,
        L=modelparams['summ_L'],
        H=modelparams['summ_H'],
        Output_dim=modelparams['summ_out'],
        num_measurements=modelparams['num_measurements'],
    )

    # load in weights for both models
    model_tr.load_state_dict(
        torch.load(filepath_FC, map_location=torch.device('cpu')), strict=True
    )
    model_tr.eval()
    model_summ_tr.load_state_dict(
        torch.load(filepath_Summ, map_location=torch.device('cpu')),
        strict=True,
    )
    model_summ_tr.eval()

    return model_tr, model_summ_tr


def get_observation(
    designs: numpy.ndarray,
    true_m: int,
    true_theta: List[float] = [
        0.7,
        0.7,
        1,
        0.3,
        0.1,
        0.7,
        0.1,
        0.5,
        0.5,
        0.9,
        0.1,
    ],
    num_arms: int = 3,
    num_trials: int = 30,
    num_blocks: int = 2,
) -> float:
    """Generate a set of observations using the specified ground-truth parameters."""

    # prepare ground truth parameter
    truth = numpy.hstack((true_m, true_theta))
    if true_m == 0:
        truth[4:] = 0.0
    elif true_m == 1:
        truth[1:4] = 0.0
        truth[6:] = 0.0
    elif true_m == 2:
        truth[1:6] = 0.0

    # simulate data
    designs_reshaped = designs.reshape(num_blocks, num_arms)
    y_obs = torch.empty(size=(1, num_blocks, 2 * num_trials), device=device)
    for i in range(len(designs_reshaped)):
        d = designs_reshaped[i].reshape(-1)
        y = simulate_bandit_batch(
            model_params=truth.reshape(1, -1),
            d=d,
            num_trials=num_trials,
            num_arms=num_arms,
            batch_size=1,
            n_cores=1,
            simbar=False,
        )
        y_obs[
            :,
            i,
            :,
        ] = y.reshape(1, -1)

    return y_obs.float()


def get_md_posteriors(
    d: numpy.ndarray,
    model: FullyConnected,
    model_summ: CAT_NSS,
    num: int = 10000,
) -> numpy.ndarray:
    """
    Use the provided models to compute posteriors in the MD task. Observations are generated using
    ground-truth parameters that have been carefully chosen to be realistic.
    """
    # go through each possible ground-truth, i.e. WSLTS, AEG or GLS being the true data-generator
    post_truths = list()
    for true_m in range(3):
        # repeat several times
        post = list()
        for _ in tqdm(range(num)):

            # real-world observation; the true parameters have been carefully chosen to be realistic
            y_obs = get_observation(
                d,
                true_m=true_m,
                true_theta=[
                    0.7,
                    0.7,
                    1,
                    0.3,
                    0.1,
                    0.7,
                    0.1,
                    0.5,
                    0.5,
                    0.9,
                    0.1,
                ],
                num_arms=3,
                num_trials=30,
                num_blocks=2,
            )

            # compute summary statistics using the model_summ model
            Sy_obs = model_summ(y_obs)

            # convert to tensors
            X = torch.tensor(
                numpy.arange(0, 3).reshape(-1, 1),
                dtype=torch.float,
                device=device,
            )
            X.to(X)
            Y = torch.cat(len(X) * [Sy_obs])
            Y.to(device)

            # Compute importance weights
            T = model(X, Y).data.numpy().reshape(-1)
            prior_weight = 1 / 3.0
            post_weights = numpy.exp(T - 1) * prior_weight
            post_norm = post_weights / numpy.sum(post_weights)
            post.append(post_norm)
        post_truths.append(post)
    post_truths = numpy.array(post_truths)

    return post_truths


def get_pe_posteriors_histograms(
    d: numpy.ndarray,
    model: FullyConnected,
    model_summ: CAT_NSS,
    prior_samples: numpy.ndarray,
    bins_all: List[numpy.ndarray],
    simmodel: int = 0,
    num_repeats: int = 1000,
    num_resamples: int = 1_000_000,
) -> Tuple[numpy.ndarray, List[numpy.ndarray]]:

    histvals_repeat = list()
    for _ in tqdm(range(num_repeats)):

        # real-world observation
        y_obs = get_observation(
            d,
            true_m=simmodel,
            true_theta=[0.7, 0.7, 0.3, 0.3, 0.1, 0.7, 0.1, 0.5, 0.5, 0.9, 0.1],
            num_arms=3,
            num_trials=30,
            num_blocks=3,
        )

        # compute summary statistics
        Sy_obs = model_summ(y_obs)

        # select parameters according to the correct model
        if simmodel == 0:
            prior_select = prior_samples[:, 1:4]
        elif simmodel == 1:
            prior_select = prior_samples[:, 4:6]
        elif simmodel == 2:
            prior_select = prior_samples[:, 6:-1]

        # convert to tensors
        X = torch.tensor(prior_select, dtype=torch.float, device=device)
        X.to(X)
        Y = torch.cat(len(X) * [Sy_obs])
        Y.to(device)

        # Compute importance weights
        T = model(X, Y.float()).data.numpy().reshape(-1)
        post_weights = numpy.exp(T - 1)
        ws_norm = post_weights / numpy.sum(post_weights)

        # get posterior samples
        idx_samples = random.choices(
            range(len(ws_norm)), weights=ws_norm, k=num_resamples
        )
        post_samples = prior_select[idx_samples]

        # get value of histograms
        hist_tmp = list()
        for idx in range(post_samples.shape[-1]):
            p = post_samples[:, idx]
            hist, _ = numpy.histogram(p, bins=bins_all[idx], density=True)
            hist_tmp.append(hist.tolist())
        histvals_repeat.append(hist_tmp)
    histvals_repeat = numpy.array(histvals_repeat)

    return histvals_repeat, bins_all


def get_pe_posteriors_densities(
    histograms: numpy.ndarray,
    bins: List[numpy.ndarray],
    prior_samples: numpy.ndarray,
    simmodel: int = 0,
    grid_size: int = 100,
    resample_size: int = 10_000,
) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:

    densities = list()
    spaces = list()
    for idx in range(histograms.shape[1]):

        # computing the KDE
        param_range = [numpy.min(bins[idx]), numpy.max(bins[idx])]
        center = (bins[idx][:-1] + bins[idx][1:]) / 2
        width = bins[idx][1] - bins[idx][0]
        x = numpy.linspace(param_range[0], param_range[-1], grid_size)
        spaces.append(x)

        # get prior samples
        prior_select = prior_samples[
            numpy.where(prior_samples[:, 0] == simmodel)
        ]

        # compute mean
        hist_mean = numpy.mean(histograms[:, idx, :], axis=0)

        # specify KDEs
        hn = hist_mean / numpy.sum(hist_mean)
        res = numpy.random.choice(center, p=hn, size=resample_size)
        kde = sts.gaussian_kde(res, bw_method='scott')

        # compute density
        density = kde.pdf(x)
        densities.append(density)

    return densities, spaces


def plot_confusion_matrix_post(
    posts: numpy.ndarray,
    fig: plt.Figure,
    ax: plt.Axes,
    num_decimals: int = 3,
    colorbar: bool = True,
    numsize: int = 20,
    cmap: str = 'Blues',
    mode: str = 'conf',
) -> Tuple[
    plt.Figure, plt.Axes, plt.Axes.imshow, numpy.ndarray, numpy.ndarray
]:
    """Using the provided MD posteriors, compute confusion matrices and plot them."""

    post_mean = numpy.mean(posts, axis=1)
    preds = numpy.argmax(posts, axis=2)

    conf = list()
    for p in preds:
        hit = [numpy.sum((p == i) * 1) for i in range(len(preds))]
        conf.append(hit)
    conf = numpy.array(conf)
    conf = conf / numpy.sum(conf, axis=1)

    if mode == 'conf':
        plot = conf
    elif mode == 'post':
        plot = post_mean

    size = 3

    # Limits for the extent
    x_start = 0
    x_end = 3
    y_start = 0
    y_end = 3

    extent = [x_start, x_end, y_start, y_end]

    plot = plot[::-1]
    im = ax.imshow(
        plot,
        origin='lower',
        extent=extent,
        interpolation=None,
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    # Add the text
    jump_x = (x_end - x_start) / (2.0 * size)
    jump_y = (y_end - y_start) / (2.0 * size)
    x_positions = numpy.linspace(
        start=x_start, stop=x_end, num=size, endpoint=False
    )
    y_positions = numpy.linspace(
        start=y_start, stop=y_end, num=size, endpoint=False
    )

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = plot[y_index, x_index]
            text_x = x + jump_x
            text_y = y + jump_y
            if label < 0.6:
                color = 'black'
            else:
                color = 'white'
            label = str(label)[: 2 + num_decimals]
            ax.text(
                text_x,
                text_y,
                label,
                color=color,
                ha='center',
                va='center',
                size=numsize,
            )

    if colorbar:
        fig.colorbar(im, fraction=0.046, pad=0.04)

    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_yticks([0.5, 1.5, 2.5])

    return fig, ax, im, post_mean, conf
