"""Tabular Monte Carlo control for three terminal-wealth goals.

One learner trades once per environment step with a uniformly sampled opponent.
Other agents trade amongst themselves in the same step, at a fixed stake.
Each episode restarts everyone with equal wealth. The learner chooses its wager.
Terminal-only reward is propagated to visited state/actions via Monte Carlo updates.
"""
from dataclasses import dataclass
import numpy as np
from .economy import trade

GOALS = {
    "Gambler": "Finish in the top 1%",
    "Comfortable": "Finish strictly above the median",
    "Survivor": "Avoid the bottom 10%",
}
ACTIONS = np.array([0.0, .02, .05, .10, .20, .40, .80, 1.0])
# Log-scale relative wealth bins; coarse progress bins keep the state space small.
RATIO_BINS = np.array([.1, .25, .5, .8, 1.0, 1.3, 2., 4., 10.])
PROGRESS_BINS = np.array([.25, .50, .75])

@dataclass
class Trained:
    name: str
    q: np.ndarray
    curve: np.ndarray

def state_of(w, steps_done, horizon):
    median = max(float(np.median(w)), 1e-12)
    ratio = float(w[0]) / median
    return (int(np.digitize(ratio, RATIO_BINS)),
            int(np.digitize(steps_done / horizon, PROGRESS_BINS)))

def succeeded(w, goal):
    """Ranks handle ties symmetrically using strict/weak thresholds."""
    x = w[0]
    n = len(w)
    if goal == "Gambler":
        # Top ceil(1% of N) places; ties at boundary count as success.
        return int(np.sum(w > x) < max(1, int(np.ceil(.01*n))))
    if goal == "Comfortable":
        return int(x > np.median(w))
    if goal == "Survivor":
        # Bottom ceil(10% of N) ranks fail, ties at boundary are conservative.
        return int(np.sum(w < x) >= max(1, int(np.ceil(.1*n))))
    raise ValueError(goal)

def episode(q, goal, n, horizon, initial, other_bet, rng,
            epsilon=0.0, learn=False, alpha=.08,
            forced_action=None, record=False):
    w = np.full(n, float(initial))
    visited, path, bets = [], [w[0]], []
    for step in range(horizon):
        state = state_of(w, step, horizon)
        if forced_action == "baseline":
            action = int(np.argmin(np.abs(ACTIONS - other_bet)))
        elif forced_action == "random":
            action = int(rng.integers(len(ACTIONS)))
        elif rng.random() < epsilon:
            action = int(rng.integers(len(ACTIONS)))
        else:
            # Random tie-break among best actions avoids fixed first-action bias.
            values = q[state]
            best = np.flatnonzero(np.isclose(values, values.max()))
            action = int(rng.choice(best))
        visited.append((*state, action))
        bets.append(ACTIONS[action])
        j = int(rng.integers(1, n))
        trade(w, 0, j, ACTIONS[action], rng)
        # One non-learner trade per step; all non-learners use the same rule.
        i = int(rng.integers(1, n))
        j = int(rng.integers(1, n-1))
        if j >= i: j += 1
        trade(w, i, j, other_bet, rng)
        if record: path.append(w[0])
    reward = succeeded(w, goal)
    if learn:
        # Every visited state/action receives the episode's terminal reward.
        for idx in set(visited):
            q[idx] += alpha * (reward - q[idx])
    return reward, np.asarray(path), np.asarray(bets)

def train_all(n=100, horizon=400, episodes=600, initial=100.,
              other_bet=.2, seed=42, allow_skip=True):
    if n < 10 or horizon < 10 or episodes < 1:
        raise ValueError("Need >=10 agents, >=10 rounds, and >=1 episode")
    rng = np.random.default_rng(int(seed))
    result = {}
    for goal in GOALS:
        q = np.zeros((len(RATIO_BINS)+1, len(PROGRESS_BINS)+1, len(ACTIONS)))
        if not allow_skip:
            q[..., 0] = -1e6
        rewards = []
        for ep in range(episodes):
            epsilon = max(.05, .45 * (1 - ep / episodes))
            reward, _, _ = episode(q, goal, n, horizon, initial,
                                   other_bet, rng, epsilon=epsilon, learn=True)
            rewards.append(reward)
        # Smooth raw training success over a trailing window.
        window = max(10, episodes//20)
        curve = np.array([np.mean(rewards[max(0,i-window+1):i+1])
                          for i in range(episodes)])
        result[goal] = Trained(goal, q, curve)
    return result

def evaluate(models, n=100, horizon=400, initial=100., other_bet=.2,
             seed=123, trials=150):
    """Independent evaluation episodes, separate RNG streams for policies."""
    results = {}
    for k, (goal, model) in enumerate(models.items()):
        rng_rl = np.random.default_rng(seed + 1000*k)
        rng_random = np.random.default_rng(seed + 1000*k + 1)
        rl, random, paths, bets = [], [], [], []
        for i in range(trials):
            r, path, actions = episode(model.q, goal, n, horizon, initial,
                                        other_bet, rng_rl, record=(i < 25))
            b, _, _ = episode(model.q, goal, n, horizon, initial,
                               other_bet, rng_random, forced_action="baseline")
            rl.append(r); random.append(b)
            if i < 25: paths.append(path); bets.append(actions)
        results[goal] = dict(rl=float(np.mean(rl)), random=float(np.mean(random)),
                             rl_successes=int(sum(rl)), random_successes=int(sum(random)),
                             trials=trials, paths=np.asarray(paths),
                             bets=np.asarray(bets))
    return results
