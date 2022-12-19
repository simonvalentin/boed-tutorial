import numpy as np
import torch
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
from scipy.special import softmax
from scipy.stats import truncnorm
import copy as copy
np.seterr(divide='ignore')


class BernoulliBandit(object):
    def __init__(self, reward_ps, num_arms=2):
        """
        Parameters:
        -----------
        num_arms: int
            Number of arms.
        reward_ps: list[float]
            Reward probabilities for arms.
        """
        self.num_arms = num_arms
        self.reward_ps = reward_ps
        self.best_p = max(self.reward_ps)  # for reference computations

    def sample_reward(self, a):
        return np.random.binomial(1, self.reward_ps[a])


class AgentModel(object):
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.counts = [0] * num_arms
        self.actions = []


class EpsilonWSLS(AgentModel):
    def __init__(self, num_arms, eps):
        """
        Parameters:
        -----------
        num_arms: int
            Number of bandit arms.
        eps: float
            The probability to stay after winning and to switch after loosing.
        """
        super(EpsilonWSLS, self).__init__(num_arms)
        self.eps = eps
        self.prev_a = np.random.randint(
            0, self.num_arms
        )  # choose initial action at random
        # Chosen arbitrarily to be a reward.
        self.prev_r = 1

    def explore(self):
        arms = np.setdiff1d(np.arange(0, self.num_arms), np.array([self.prev_a]))
        return np.random.choice(arms)

    def run_one_step(self, bandit):
        if self.prev_r == 1:
            if np.random.binomial(1, self.eps) == 1:
                # Stay after winning
                a = self.prev_a
            else:
                a = self.explore()
        else:
            if np.random.binomial(1, self.eps) == 1:
                # Shift/sample after loosing
                a = self.explore()
            else:
                a = self.prev_a
        r = bandit.sample_reward(a)
        self.prev_a = a
        self.prev_r = r
        return a, r
    
    
class WSLTS(AgentModel):
    def __init__(self, num_arms, gamma_w, gamma_l, temp):
        """
        Extended WSLS with Thompson-Sampling.
        
        Parameters:
        -----------
        num_arms: int
            Number of bandit arms.
        gamma_w: float
            The probability to stay after winning.
        gamma_l: float
            The probability of switching after losing.
        temp: float
            Softmax temperature. t -> 0: we recovers TS except for the prev arm; t->inf recovers WSLS.
        """
        super(WSLTS, self).__init__(num_arms)
        self.gamma_w = gamma_w
        self.gamma_l = gamma_l        
        self.temp = temp
        self.prev_a = np.random.randint(
            0, self.num_arms
        )  # choose initial action at random
        # Chosen arbitrarily to be a reward.
        self.prev_r = 1
        
        # Initialise beta distribution parameters, assume uniform prior.
        self.alphas = np.array([1] * num_arms)
        self.betas = np.array([1] * num_arms)
        

    def explore(self):
        alphas_mod = self.alphas**(1/self.temp)
        betas_mod = self.betas**(1/self.temp)
        theta_sample = np.random.beta(alphas_mod, betas_mod)
        theta_sample[self.prev_a] = 0  # previous arm will not be selected
        a = np.argmax(theta_sample)
        return a

    def run_one_step(self, bandit):
        if self.prev_r == 1:
            if np.random.binomial(1, self.gamma_w) == 1:
                # Stay after winning
                a = self.prev_a
            else:
                a = self.explore()
        else:
            if np.random.binomial(1, self.gamma_l) == 1:
                # Shift after loosing
                a = self.explore()
            else:
                a = self.prev_a
        r = bandit.sample_reward(a)
        self.prev_a = a
        self.prev_r = r
        
        self.alphas[a] += r 
        self.betas[a] += 1-r 
        return a, r


class EpsilonGreedy(AgentModel):
    def __init__(self, num_arms, eps, init_count=1):
        """
        Parameters:
        -----------
        num_arms: int
            Number of bandit arms.
        eps: float
            The probability to explore at each time step.
        init_count: int
            Defaults to optimistic initialization, has to be >= 1.
        """
        super(EpsilonGreedy, self).__init__(num_arms)
        assert init_count >= 1, "Initial pseudo-counts need to be >= 1."
        self.eps = eps
        self.counts = np.array([init_count] * num_arms)
        self.counts_reward = np.array([init_count] * num_arms)
        self.estimates = self.counts_reward / self.counts

    def run_one_step(self, bandit):
        if np.random.binomial(1, self.eps) == 1:
            # Explore uniformly at random
            a = np.random.randint(0, self.num_arms)
        else:
            # Select randomly from the argmaxes
            a = np.random.choice(
                np.argwhere(self.estimates == np.amax(self.estimates)).flatten()
            )

        self.actions.append(a)
        r = bandit.sample_reward(a)
        self.counts[a] += 1
        self.counts_reward[a] += r
        self.estimates = self.counts_reward / self.counts

        return a, r

class AutoregEGreedy(AgentModel):
    def __init__(self, num_arms, eps, lambd, init_count=1):
        """
        Autoregressive epsilon-greedy model.
        
        Parameters:
        -----------
        num_arms: int
            Number of bandit arms.
        eps: float \in [-1, 1]
            The probability to explore at each time step.
        lambd: float \in [0, 1]
            Probability of selecting the previous arm.
        init_count: int
            Defaults to optimistic initialization, has to be >= 1.
        """
        super(AutoregEGreedy, self).__init__(num_arms)
        assert init_count >= 1, "Initial pseudo-counts need to be >= 1."
        self.eps = eps
        self.lambd = lambd
        self.counts = np.array([init_count] * num_arms)
        self.counts_reward = np.array([init_count] * num_arms)
        self.estimates = self.counts_reward / self.counts
        self.prev_a = np.random.randint(0, self.num_arms)

    def run_one_step(self, bandit):

        if np.random.binomial(1, self.eps) == 1:  
            # Explore 
            p_prev = self.lambd + (1-self.lambd)/self.num_arms 
            if np.random.binomial(1, p_prev) == 1:
                a = self.prev_a
            else:
                other_arms = np.setdiff1d(np.arange(0, self.num_arms), np.array([self.prev_a]))    
                a = np.random.choice(other_arms) 
                
        else: 
            # Greedy action, select from argmaxes
            argmaxes = np.argwhere(self.estimates == np.amax(self.estimates)).flatten()
            if self.prev_a in argmaxes and len(argmaxes) > 1:
                p_prev = self.lambd + (1-self.lambd)/len(argmaxes)
                
                if np.random.binomial(1, p_prev) == 1: 
                    a = self.prev_a
                else: 
                    other_arms = np.setdiff1d(argmaxes, np.array([self.prev_a]))
                    a = np.random.choice(other_arms) 
            else: 
                a = np.random.choice(argmaxes)

        self.actions.append(a)
        r = bandit.sample_reward(a)
        self.counts[a] += 1
        self.counts_reward[a] += r
        self.estimates = self.counts_reward / self.counts
        self.prev_a = a

        return a, r


class GLS(AgentModel):
    def __init__(self, num_arms, gamma, transition, initial_state_mu=0.5, return_latent=False):
        """
        Parameters:
        -----------
        num_arms: int
            Number of bandit arms.
        gamma: float
            Probability of following deterministic heuristic, 'Accuracy of execution'.
        transition: np.array
            Transition model for latent state.
        initial_state_mu: float
            Probability of being in an exploit state initially.
        """
        super(GLS, self).__init__(num_arms)
        self.gamma = gamma
        self.transition = transition
        self.counts = np.array([0] * num_arms)
        self.counts_reward = np.array([0] * num_arms)
        self.counts_failures = np.array([0] * num_arms)
        self.latent_state = np.random.binomial(1, initial_state_mu)
        self.prev_latent_state = copy.copy(self.latent_state)
        self.return_latent = return_latent
               

    def run_one_step(self, bandit):
        self.max_rewards = np.argwhere(
            self.counts_reward == np.amax(self.counts_reward)
        ).flatten()
        self.min_failures = np.argwhere(
            self.counts_failures == np.amin(self.counts_failures)
        ).flatten()
        
        

        same_arms = np.intersect1d(self.max_rewards, self.min_failures)
        self.same_arms = same_arms
        if len(same_arms) > 1:
            a = np.random.choice(same_arms)

        elif len(same_arms) == 1:
            if np.random.binomial(1, self.gamma):
                a = same_arms[0]
            else:
                other_arms = np.setdiff1d(np.arange(0, self.num_arms), same_arms)
                a = np.random.choice(other_arms)

        else:
            self.prev_latent_state = copy.copy(self.latent_state)
            # Latent state depends on previous latent state and on previous reward/loss
            self.latent_state = np.random.binomial(1, 
                                                   self.transition[self.latent_state,self.prev_r])
            if self.latent_state == 0:
                explore_arms = [
                    arm
                    if self.counts_reward[arm]
                    == np.amax(self.counts_reward[self.min_failures])
                    else None
                    for arm in self.min_failures
                ]
                explore_arms = np.array([x for x in explore_arms if x is not None])

                if np.random.binomial(1, self.gamma):
                    # choose search arm
                    a = np.random.choice(explore_arms)
                else:
                    other_arms = np.setdiff1d(
                        np.arange(0, self.num_arms), np.array(explore_arms)
                    )
                    a = np.random.choice(other_arms)
            else:
                exploit_arms = [
                    arm
                    if self.counts_failures[arm]
                    == np.amin(self.counts_failures[self.max_rewards])
                    else None
                    for arm in self.max_rewards
                ]
                exploit_arms = np.array([x for x in exploit_arms if x is not None])

                if np.random.binomial(1, self.gamma):
                    a = np.random.choice(exploit_arms)

                else:
                    other_arms = np.setdiff1d(
                        np.arange(0, self.num_arms), np.array(exploit_arms)
                    )
                    a = np.random.choice(other_arms)
        self.actions.append(a)
        r = bandit.sample_reward(a)
        self.prev_r = r
        self.counts[a] += 1
        self.counts_reward[a] += r
        self.counts_failures = self.counts - self.counts_reward
        
        if self.return_latent:  
            return a, r, self.prev_latent_state
        else: 
            return a, r

def run_bandit(bandit, agent_model, num_trials=30):
    choices = []
    rewards = []
    for t in range(num_trials):
        a, r = agent_model.run_one_step(bandit)
        choices.append(a)
        rewards.append(r)
    return choices, rewards

def run_latent_bandit(bandit, agent_model, num_trials=30):
    choices = []
    rewards = []
    latents = []
    for t in range(num_trials):
        a, r, l = agent_model.run_one_step(bandit)
        choices.append(a)
        rewards.append(r)
        latents.append(l)
    return choices, rewards, latents


def run_wsls(model_param, d, num_trials, num_arms):
    """
    Parameters:
    -----------

    model_params: list
        List containing epsilon value.
    d: list
        List of design variables, containing reward probabilites.
    num_trials: int
        Number of trials.
    num_arms: int
        Number of bandit arms.
    """
    bandit = BernoulliBandit(reward_ps=d, num_arms=num_arms)
    agent_model = EpsilonWSLS(num_arms=num_arms, eps=model_param)
    cc, rr = run_bandit(bandit, agent_model, num_trials)
    return torch.from_numpy(np.row_stack((cc, rr)))

def run_wslts(model_params, d, num_trials, num_arms):
    """
    Parameters:
    -----------

    model_params: list
        List containing gamma values and temperature.
    d: list
        List of design variables, containing reward probabilites.
    num_trials: int
        Number of trials.
    num_arms: int
        Number of bandit arms.
    """
    bandit = BernoulliBandit(reward_ps=d, num_arms=num_arms)
    agent_model = WSLTS(num_arms=num_arms, gamma_w=model_params[0],
                              gamma_l=model_params[1], temp=model_params[2])
    cc, rr = run_bandit(bandit, agent_model, num_trials)
    return torch.from_numpy(np.row_stack((cc, rr)))


def run_epsgreedy(model_param, d, num_trials, num_arms):
    """
    Parameters:
    -----------

    model_params: list
        List containing epsilon value.
    d: list
        List of design variables, containing reward probabilites.
    num_trials: int
        Number of trials.
    num_arms: int
        Number of bandit arms.
    """
    bandit = BernoulliBandit(reward_ps=d, num_arms=num_arms)
    agent_model = EpsilonGreedy(num_arms=num_arms, eps=model_param)
    cc, rr = run_bandit(bandit, agent_model, num_trials)
    return torch.from_numpy(np.row_stack((cc, rr)))


def run_autoregegreedy(model_params, d, num_trials, num_arms):
    """
    Parameters:
    -----------

    model_params: list
        List containing epsilon value and lambda.
    d: list
        List of design variables, containing reward probabilites.
    num_trials: int
        Number of trials.
    num_arms: int
        Number of bandit arms.
    """
    bandit = BernoulliBandit(reward_ps=d, num_arms=num_arms)
    agent_model = AutoregEGreedy(num_arms=num_arms, eps=model_params[0], 
                                 lambd=model_params[1])
    cc, rr = run_bandit(bandit, agent_model, num_trials)
    return torch.from_numpy(np.row_stack((cc, rr)))

def run_gls(model_params, d, num_trials, num_arms):
    """
    Parameters:
    -----------

    model_params: list
        List containing gamma value and state probability.
    d: list
        List of design variables, containing reward probabilites.
    num_trials: int
        Number of trials.
    num_arms: int
        Number of bandit arms.
    """
    bandit = BernoulliBandit(reward_ps=d, num_arms=num_arms)
    agent_model = GLS(
        num_arms=num_arms, gamma=model_params[0], transition=model_params[1:5].reshape(2, 2),
        initial_state_mu=model_params[5]
    )
    cc, rr = run_bandit(bandit, agent_model, num_trials)
    return torch.from_numpy(np.row_stack((cc, rr)))

def run_gls_latent(model_params, d, num_trials, num_arms):
    """
    Parameters:
    -----------

    model_params: list
        List containing gamma value and state probability.
    d: list
        List of design variables, containing reward probabilites.
    num_trials: int
        Number of trials.
    num_arms: int
        Number of bandit arms.
    """
    bandit = BernoulliBandit(reward_ps=d, num_arms=num_arms)
    agent_model = GLS(
        num_arms=num_arms, gamma=model_params[0], transition=model_params[1:5].reshape(2, 2),
        initial_state_mu=model_params[5], return_latent=True
    )
    cc, rr, ll = run_latent_bandit(bandit, agent_model, num_trials)
    return torch.from_numpy(np.row_stack((cc, rr, ll)))


def simulate_bandit(model_params, d, num_trials, num_arms):
    """
    Parameters:
    -----------
    model_params: list
        List containing model indicator, then parameters, formatted as [m, wslts, autoregegreedy, gls]
    d: list
        List of design variables, containing reward probabilites.
    num_trials: int
        Number of trials.
    num_arms: int
        Number of bandit arms.

    Returns
    -------
    data: torch.tensor
        Vector, where format is [action_1, action_2, ...., reward_1, reward_2, ...].

    """
    if model_params[0] == 0:
        data = run_wslts(model_params[1:4], d, num_trials, num_arms)
    elif model_params[0] == 1:
        data = run_autoregegreedy(model_params[4:6], d, num_trials, num_arms)
    elif model_params[0] == 2:
        data = run_gls(model_params[6:], d, num_trials, num_arms)
    else:
        raise NotImplementedError("Model indicator calls undefined model.")
    return data


def simulate_bandit_batch(
    model_params, d, num_trials, num_arms, batch_size, n_cores=1, simbar=True
):
    """
    Parameters:
    -----------
    model_params: np.array
        Numpy array containing model indicator and parameters as rows, each
        formatted as [m, wsls_eps, egreedy_eps].
    d: list
        List of design variables, containing reward probabilites.
    num_trials: int
        Number of trials.
    num_arms: int
        Number of bandit arms.
    batch_size: int
        Number of sequences to sample.
    n_cores: int
        Number of cores used for running simulator in parallel.

    Returns
    -------
    data: list[torch.tensor]
        List of data vectors.

    """
    if n_cores == 1:
        batch = []
        for i in tqdm(range(batch_size), disable=not simbar):
            batch.append(simulate_bandit(model_params[i, :], d, num_trials, num_arms))
    elif n_cores > 1:
        batch = Parallel(n_jobs=n_cores)(
            delayed(simulate_bandit)(model_params[i, :], d, num_trials, num_arms)
            for i in tqdm(range(batch_size), disable=not simbar)
        )
    else:
        raise ValueError("Invalid number of cores specified.")
    return torch.stack(batch)


def sim_bandit_prior(batch_size, prior="empirical", simmodel=None):
    """
    Priors for WSLS and e-greedy, using empirically inferred priors from Zhang and Lee (2010) for WSLS
    and e-greedy, prior for latent state uninformed.
    
    model: Bool or int
        Set model=None for MD or MDPE task and set it to 0, 1, 2 for PE tasks
        for the corresponding model.
            
    """
    # Assumes three models.
    priors = np.zeros((batch_size, 12))

    if simmodel is None:
        # Sample model indicator, assuming a uniform prior
        priors[:, 0] = np.random.choice([0, 1, 2], batch_size)
    else:
        # constant model indicator, model in {0, 1, 2}
        priors[:, 0] = np.ones(batch_size) * simmodel

    if prior == "informative":

        # Sample gamma_w for WSLS
        priors[:, 1] = np.random.uniform(0.5, 1, batch_size)    
        
        # Sample gamma_l for WSLSs
        priors[:, 2] = np.random.uniform(0.5, 1, batch_size)

        # Sample softmax temperature for WSLS
        priors[:, 3] = np.random.lognormal(0, 1, batch_size)  
        
        # Sample epsilon for autoregressive e-greedy
        priors[:, 4] = np.random.uniform(0, 1, batch_size)
        
        # Sample lambda for autoregressive e-greedy
        priors[:, 5] = np.random.uniform(0, 1, batch_size)
        
        # Sample gamma for latent state
        priors[:, 6] = np.random.uniform(0.5, 1, batch_size)    

        # Sample explore state probabilities for latent state
        priors[:, 7] = np.random.uniform(0, 1, batch_size)    
        priors[:, 8] = np.random.uniform(0, 1, batch_size)    
        priors[:, 9] = np.random.uniform(0, 1, batch_size)
        priors[:, 10] = np.random.uniform(0, 1, batch_size)    
        
        # Set initial probability for latent state, fixed
        priors[:, 11] = 0.0 # np.random.uniform(0, 1, batch_size)

    elif prior == "uninformed":
        # Sample gamma_w for WSLS
        priors[:, 1] = np.random.beta(1, 1, batch_size)
        
        # Sample gamma_l for WSLS
        priors[:, 2] = np.random.beta(1, 1, batch_size)
        
        # Sample softmax temperature for WSLS
        priors[:, 3] = np.random.lognormal(0, 1, batch_size) 
        
        # Sample epsilon for autoregressive e-greedy
        priors[:, 4] = np.random.beta(1, 1, batch_size)
        
        # Sample lambda for autoregressive e-greedy
        priors[:, 5] = np.random.beta(1, 1, batch_size)
        
        # Sample gamma for latent state
        priors[:, 6] = np.random.beta(1, 1, batch_size)

        # Sample state probability for latent state
        priors[:, 7:11] = np.random.beta(1, 1, (batch_size, 4))
        
        # Set initial probability for latent state, fixed
        priors[:, 11] = 0.5

    else:
        raise ValueError("Prior not implemented.")

    for p in priors:
        if p[0] == 0:
            p[4:] = 0.0
        elif p[0] == 1:
            p[1:4] = 0.0
            p[6:] = 0.0
        elif p[0] == 2:
            p[1:6] = 0.0

    return priors

    

class BanditDatasetMD_Multiple(torch.utils.data.Dataset):
    def __init__(
        self, designs, prior, device,
        num_trials=15, num_blocks=1, num_arms=2,
        n_cores=1, simbar=True):

        """
        A bandits model dataset for model discrimination.
        Parameters
        ----------
        designs: torch.tensor, numpy array or list
            Design variables that we want to optimise.
        prior: numpy array
            Samples from the prior distribution.
        device: torch.device
            Device to run the training process on.
        """
        super(BanditDatasetMD_Multiple, self).__init__()

        # prior samples to PyTorch tensors
        X = torch.tensor(prior, dtype=torch.float, device=device, requires_grad=False)
        self.m = X[:, 0].reshape(-1, 1)
        
        # simulate data
        designs_reshaped = designs.reshape(num_blocks, num_arms)
        self.Y = torch.empty(
            size=(len(self.m), num_blocks, 2 * num_trials), device=device)

        for i in range(len(designs_reshaped)):
            d = designs_reshaped[i].reshape(-1)
            y = simulate_bandit_batch(
                model_params=prior,
                d=d,
                num_trials=num_trials,
                num_arms=num_arms,
                batch_size=len(self.m),
                n_cores=n_cores,
                simbar=simbar,
            )
            self.Y[:,i,:,] = y.reshape(len(self.m), -1)

        # reshape data into the correct format (batch_size, -1)
        # also convert ints to floats (to work with NNs)
        self.Y = self.Y.float()

    def __getitem__(self, idx):
        """ Get Prior samples and data by index.
        Parameters
        ----------
        idx : int
            Item index
        Returns
        -------
        Batched prior samples, batched data samples
        """
        return self.m[idx], self.Y[idx]

    def __len__(self):
        """Number of samples in the dataset"""
        return len(self.m)
    
    
class BanditDatasetPE_Multiple(torch.utils.data.Dataset):
    def __init__(
        self, designs, prior, device,
        num_trials=15, num_blocks=1, num_arms=2,
        n_cores=1, simbar=True, simmodel=0):

        """
        A bandits model dataset for parameter estimation at multiple designs.
        Parameters
        ----------
        designs: torch.tensor, numpy array or list
            Design variables that we want to optimise.
        prior: numpy array
            Samples from the prior distribution.
        device: torch.device
            Device to run the training process on.
        """

        super(BanditDatasetPE_Multiple, self).__init__()

        # select the appropriate prior samples
        if simmodel == 0:  # WSLS
            self.X = torch.tensor(
                prior[:,1:4], dtype=torch.float, device=device, requires_grad=False)
        elif simmodel == 1:  # E-Greedy
            self.X = torch.tensor(
                prior[:,4:6], dtype=torch.float, device=device, requires_grad=False)
        elif simmodel == 2:  # GLS
            # Note: we can ignore the last parameter because it's fixed! This is not used
            # for data simulation anyway.
            self.X = torch.tensor(
                prior[:,6:11], dtype=torch.float, device=device, requires_grad=False)

        # simulate data
        designs_reshaped = designs.reshape(num_blocks, num_arms)
        self.Y = torch.empty(
            size=(len(self.X), num_blocks, 2 * num_trials), device=device)

        for i in range(len(designs_reshaped)):
            d = designs_reshaped[i].reshape(-1)
            y = simulate_bandit_batch(
                model_params=prior,
                d=d,
                num_trials=num_trials,
                num_arms=num_arms,
                batch_size=len(self.X),
                n_cores=n_cores,
                simbar=simbar,
            )
            self.Y[:,i,:,] = y.reshape(len(self.X), -1)

        # reshape data into the correct format (batch_size, -1)
        # also convert ints to floats (to work with NNs)
        self.Y = self.Y.float()

    def __getitem__(self, idx):
        """ Get Prior samples and data by index.
        Parameters
        ----------
        idx : int
            Item index
        Returns
        -------
        Batched prior samples, batched data samples
        """
        return self.X[idx], self.Y[idx]

    def __len__(self):
        """Number of samples in the dataset"""
        return len(self.X)
