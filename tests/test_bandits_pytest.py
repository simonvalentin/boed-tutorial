import pytest
from boed.simulators.bandits import *


def test_wsls_deterministic():
    bandit = BernoulliBandit(reward_ps=[0.0, 1.0], num_arms=2)
    agent_model = WSLTS(num_arms=2, gamma_w=1.0, gamma_l=1.0, temp=10**10)
    agent_model.prev_a = 1
    agent_model.prev_r = 1
    cc, rr = run_bandit(bandit, agent_model, num_trials=3)
    assert cc == [1, 1, 1], "Deterministic WSLS is incorrect"


def test_wsls_deterministic_switch():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0], num_arms=2)
    agent_model = WSLTS(num_arms=2, gamma_w=1.0, gamma_l=1.0, temp=10**10)
    agent_model.prev_a = 0
    agent_model.prev_r = 1
    cc, rr = run_bandit(bandit, agent_model, num_trials=3)
    assert cc == [0, 1, 0], "Deterministic WSLS is incorrect"


def test_wsls_deterministic_stay():
    bandit = BernoulliBandit(reward_ps=[1.0, 1.0], num_arms=2)
    agent_model = WSLTS(num_arms=2, gamma_w=1.0, gamma_l=1.0, temp=10**10)
    agent_model.prev_a = 0
    agent_model.prev_r = 1
    cc, rr = run_bandit(bandit, agent_model, num_trials=3)
    assert cc == [0, 0, 0], "Deterministic WSLS is incorrect"


def test_wsls_smalleps_switch():
    bandit = BernoulliBandit(reward_ps=[1.0, 1.0], num_arms=2)
    agent_model = WSLTS(num_arms=2, gamma_w=0.0, gamma_l=0.0, temp=10**10)
    agent_model.prev_a = 0
    agent_model.prev_r = 1
    cc, rr = run_bandit(bandit, agent_model, num_trials=3)
    assert cc == [1, 0, 1], "Deterministic WSLS is incorrect"


def test_wsls_smalleps_stay():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0], num_arms=2)
    agent_model = WSLTS(num_arms=2, gamma_w=0.0, gamma_l=0.0, temp=10**10)
    agent_model.prev_a = 1
    agent_model.prev_r = 0
    cc, rr = run_bandit(bandit, agent_model, num_trials=3)
    assert cc == [1, 1, 1], "Deterministic WSLS is incorrect"


def test_epsgreedy():
    bandit = BernoulliBandit(reward_ps=[1.0, 0.0], num_arms=2)
    agent_model = AutoregEGreedy(num_arms=2, eps=0.0, lambd=0)
    agent_model.estimates = [0.0, 0.1]
    cc, rr = run_bandit(bandit, agent_model, num_trials=1)
    assert cc == [1], "Deterministic e-greedy is incorrect"


def test_epsgreedy_estimates():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0], num_arms=2)
    agent_model = AutoregEGreedy(num_arms=2, eps=0.5, lambd=0, init_count=1)
    cc, rr = run_bandit(bandit, agent_model, num_trials=1)
    assert (agent_model.estimates == [0.5, 1.0]).all() or (
        agent_model.estimates == [1.0, 0.5]
    ).all(), "Deterministic e-greedy is incorrect"


def test_latent_same():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0], num_arms=2)
    agent_model = GLS(num_arms=2, gamma=1.0, transition=np.full((2,2),0.5))
    agent_model.counts_reward = np.array([2, 2])
    agent_model.counts_failures = np.array([1, 1])
    _, _ = run_bandit(bandit, agent_model, num_trials=1)
    assert np.array_equal(
        agent_model.same_arms, np.array([0, 1])
    ), "Latent switch same situation is incorrect"


def test_latent_better_worse():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0], num_arms=2)
    agent_model = GLS(num_arms=2, gamma=1.0, transition=np.full((2,2),0.5))
    agent_model.counts_reward = np.array([1, 0])
    agent_model.counts_failures = np.array([0, 1])
    cc, rr = run_bandit(bandit, agent_model, num_trials=1)
    assert cc == [0], "Better-worse situation handled incorrectly."


def test_latent_better_worse_reverse():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0], num_arms=2)
    agent_model = GLS(num_arms=2, gamma=0.0, transition=np.full((2,2),0.5))
    agent_model.counts_reward = np.array([1, 0])
    agent_model.counts_failures = np.array([0, 1])
    cc, rr = run_bandit(bandit, agent_model, num_trials=1)
    assert cc == [1], "Better-worse situation handled incorrectly."


def test_latent_explore_exploit():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0], num_arms=2)
    agent_model = GLS(num_arms=2, gamma=1.0, transition=np.full((2,2),1.))
    agent_model.counts_reward = np.array([3, 1])
    agent_model.counts_failures = np.array([2, 0])
    agent_model.prev_r = 1
    cc, rr = run_bandit(bandit, agent_model, num_trials=1)
    assert cc == [0], "Explore-exploit situation handled incorrectly."


def test_latent_explore_exploit_reverse():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0], num_arms=2)
    agent_model = GLS(num_arms=2, gamma=1.0, transition=np.full((2,2),0.))
    agent_model.counts_reward = np.array([3, 1])
    agent_model.counts_failures = np.array([2, 0])
    agent_model.prev_r = 1
    cc, rr = run_bandit(bandit, agent_model, num_trials=1)
    assert cc == [1], "Explore latent state incorrect."


def test_latent_explore_explore():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0], num_arms=2)
    agent_model = GLS(num_arms=2, gamma=1.0, transition=np.full((2,2),1.))
    agent_model.counts_reward = np.array([4, 1])
    agent_model.counts_failures = np.array([3, 0])
    agent_model.prev_r = 1
    cc, rr = run_bandit(bandit, agent_model, num_trials=1)
    assert cc == [0], "Explore-exploit situation handled incorrectly."


def test_latent_explore_exploit_reverse_gamma():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0], num_arms=2)
    agent_model = GLS(num_arms=2, gamma=0.0, transition=np.full((2,2),0.))
    agent_model.counts_reward = np.array([3, 1])
    agent_model.counts_failures = np.array([2, 0])
    agent_model.prev_r = 1
    cc, rr = run_bandit(bandit, agent_model, num_trials=1)
    assert cc == [0], "Explore latent state."


def test_latent_explore_exploit_threearms():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0, 0.0], num_arms=3)
    agent_model = GLS(num_arms=3, gamma=1.0, transition=np.full((2,2),1))
    agent_model.counts_reward = np.array([3, 1, 5])
    agent_model.counts_failures = np.array([2, 0, 0])
    agent_model.prev_r = 1
    cc, rr = run_bandit(bandit, agent_model, num_trials=1)
    assert cc == [2], "Explore-exploit situation handled incorrectly."

def test_latent_explore_exploit_threearms_explore():
    bandit = BernoulliBandit(reward_ps=[0.0, 0.0, 0.0], num_arms=3)
    agent_model = GLS(num_arms=3, gamma=1.0, transition=np.full((2,2),0.))
    agent_model.counts_reward = np.array([3, 3, 5])
    agent_model.counts_failures = np.array([2, 2, 4])
    agent_model.prev_r = 1
    cc, rr = run_bandit(bandit, agent_model, num_trials=1)
    assert cc == [0] or cc == [1], "Explore-exploit situation handled incorrectly."




