"""Conservative, symmetric Yard Sale trades."""
import numpy as np

def gini(wealth):
    x = np.sort(np.asarray(wealth, dtype=float))
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    n = len(x)
    return float(2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum()) - (n + 1) / n)

def trade(wealth, i, j, fraction, rng):
    """Stake is fraction of the poorer agent's wealth; each side wins with p=.5."""
    stake = float(fraction) * min(wealth[i], wealth[j])
    if rng.random() < .5:
        wealth[i] += stake
        wealth[j] -= stake
    else:
        wealth[i] -= stake
        wealth[j] += stake

def simulate(n=100, initial=100., fraction=.2, trades=100_000, seed=42, frames=101):
    n, trades = int(n), int(trades)
    if n < 2 or initial <= 0 or not 0 <= fraction <= 1 or trades < 1:
        raise ValueError("Invalid simulation settings")
    rng = np.random.default_rng(int(seed))
    w = np.full(n, float(initial))
    checkpoints = np.unique(np.linspace(0, trades, frames, dtype=int))
    snapshots, history = [], []
    def save(t):
        snapshots.append(w.copy())
        history.append((t, gini(w), w.max()/w.sum()))
    save(0)
    next_frame = 1
    for t in range(1, trades+1):
        i = int(rng.integers(n))
        j = int(rng.integers(n-1))
        if j >= i: j += 1
        trade(w, i, j, fraction, rng)
        if next_frame < len(checkpoints) and t == checkpoints[next_frame]:
            save(t)
            next_frame += 1
    if not np.isclose(w.sum(), n*initial, rtol=1e-10):
        raise AssertionError("Wealth not conserved")
    return np.asarray(history), np.asarray(snapshots)
