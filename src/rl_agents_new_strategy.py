"""Shared Yard Sale economy: Walter learns, Gustavo fixes wager, Jesse randomizes.



All three characters coexist in every episode.

The coin flip, random pairing, and conserved wealth remain unchanged.

"""

from dataclasses import dataclass

import numpy as np

try:

    from .economy import trade

except ImportError:

    from economy import trade



GOALS = {'Walter': 'Concentrator', 'Gustavo': 'Capitalist', 'Jesse': 'Grower'}

ACTIONS = np.array([0., .02, .05, .10, .20, .40, .80, 1.])

POLICIES = ('rl', 'baseline', 'random', 'aggressive')

RANK_BINS = np.array([.01, .05, .10, .25, .50, .75])

SHARE_BINS = np.array([.25, .5, 1., 2., 5., 10., 20., 25.])

OPPONENT_BINS = np.array([.1, .25, .5, 1., 2., 4., 10.])

TIME_BINS = np.array([.25, .5, .75, .9])

Q_SHAPE = (len(RANK_BINS)+1, len(SHARE_BINS)+1,

           len(OPPONENT_BINS)+1, len(TIME_BINS)+1, len(ACTIONS))



@dataclass

class Trained:

    name: str

    goal: str

    q: np.ndarray

    curve: np.ndarray

    allow_skip: bool = True

    reward_curve: np.ndarray | None = None



def state_of(w, t, horizon, opponent):

    own = float(w[0])

    richer = np.count_nonzero(w[1:] > own) / (len(w)-1)

    share_multiple = own / max(float(np.mean(w)), 1e-12)

    opponent_ratio = float(w[opponent]) / max(own, 1e-12)

    return (int(np.digitize(richer, RANK_BINS)),

            int(np.digitize(share_multiple, SHARE_BINS)),

            int(np.digitize(opponent_ratio, OPPONENT_BINS)),

            int(np.digitize(t / max(horizon, 1), TIME_BINS)))



def succeeded(w, goal, initial=100., idx=0):

    own = float(w[idx])

    if goal == 'Concentrator':

        return int(own >= .25 * float(np.sum(w)) - 1e-10)

    if goal == 'Capitalist':

        worst_rank = 1 + int(np.count_nonzero(np.delete(w, idx) >= own))

        return int(worst_rank <= max(1, int(np.ceil(.1 * len(w)))))

    if goal == 'Grower':

        return int(own >= 1.5 * initial)

    raise ValueError(goal)



def gini(w):

    x = np.sort(np.asarray(w, dtype=float))

    if x.sum() <= 0: return 0.

    n = len(x)

    return float(2 * np.dot(np.arange(1, n+1), x) / (n*x.sum()) - (n+1)/n)



def top_share(w, fraction):

    k = max(1, int(np.ceil(fraction * len(w))))

    return float(np.sum(np.sort(w)[-k:]) / np.sum(w))



def choose_action(q, state, rng, epsilon=0., allow_skip=True):

    first = 0 if allow_skip else 1

    if rng.random() < epsilon:

        return int(rng.integers(first, len(ACTIONS)))

    values = q[state][first:]

    best = np.flatnonzero(np.isclose(values, np.max(values)))

    return int(first + rng.choice(best))



def _rolling(a):

    a = np.asarray(a, dtype=float)

    window = max(10, len(a)//20)

    cumulative = np.r_[0., np.cumsum(a)]

    ends = np.arange(1, len(a)+1)

    starts = np.maximum(0, ends-window)

    return (cumulative[ends]-cumulative[starts])/(ends-starts)



def episode(q, n, horizon, initial, other_bet, rng, *,

            walter_policy='rl', learn=False, epsilon=0., alpha=.06,

            allow_skip=True, record=False):

    """One economy: Walter=0, Gustavo=1, Jesse=2, background=3...n-1."""

    w = np.full(n, float(initial))

    paths = [[float(initial)] for _ in range(3)] if record else None

    bets = [[], [], []]

    pending = None

    # Walter's terminal objective is reaching the richest 1%; also report 25% share.

    # The shaped signal is a bounded intermediate learning heuristic, not guaranteed optimal.

    def potential():

        return .2 * min(float(w[0]) / max(float(np.sum(w)) * .01, 1e-12), 5.)

    for t in range(horizon):

        i = int(rng.integers(n))

        j = int(rng.integers(n-1))

        if j >= i: j += 1

        if i == 0 or j == 0:

            opponent = j if i == 0 else i

            state = state_of(w, t, horizon, opponent)

            if learn and pending is not None:

                prev_state, action, old_potential = pending

                target = potential() - old_potential + float(np.max(q[state][0 if allow_skip else 1:]))

                key = (*prev_state, action)

                q[key] += alpha * (target - q[key])

            if walter_policy == 'rl':

                action = choose_action(q, state, rng, epsilon, allow_skip)

                fraction = float(ACTIONS[action])

            elif walter_policy == 'baseline':

                action, fraction = None, float(other_bet)

            elif walter_policy == 'random':

                action = int(rng.integers(0 if allow_skip else 1, len(ACTIONS)))

                fraction = float(ACTIONS[action])

            elif walter_policy == 'aggressive':

                action, fraction = None, 1.

            else:

                raise ValueError(walter_policy)

            if learn and action is not None:

                pending = (state, action, potential())

            bets[0].append(fraction)

        elif i == 2 or j == 2:

            fraction = float(ACTIONS[int(rng.integers(0 if allow_skip else 1, len(ACTIONS)))])

            bets[2].append(fraction)

        else:

            fraction = float(other_bet)

            if i == 1 or j == 1:

                bets[1].append(fraction)

        trade(w, i, j, fraction, rng)

        if record:

            for k in range(3): paths[k].append(float(w[k]))

    # Reward a top-1% finish (with n=100, strictly richest); not a guaranteed outcome.

    walter_success = int(1 + np.count_nonzero(w[1:] >= w[0]) <= max(1, int(np.ceil(.01*n))))

    if learn and pending is not None:

        state, action, old_potential = pending

        target = float(walter_success) + potential() - old_potential

        key = (*state, action)

        q[key] += alpha * (target - q[key])

    return {

        'wealth': w,

        'success': {name: succeeded(w, goal, initial, idx)

                    for idx, (name, goal) in enumerate(GOALS.items())},

        'walter_top1': walter_success,

        'gini': gini(w),

        'top1_share': top_share(w, .01),

        'top5_share': top_share(w, .05),

        'top10_share': top_share(w, .10),

        'paths': paths,

        'bets': bets,

    }



def train_all(n=100, horizon=5000, episodes=12000, initial=100.,

              other_bet=.2, seed=42, allow_skip=True):

    n, horizon, episodes = int(n), int(horizon), int(episodes)

    if n < 10 or horizon < 1 or episodes < 1 or initial <= 0 or not 0 <= other_bet <= 1:

        raise ValueError('Invalid experiment settings')

    q = np.zeros(Q_SHAPE, dtype=float)

    rng = np.random.default_rng(int(seed))

    history = {name: [] for name in GOALS}

    walter_top1 = []

    for ep in range(episodes):

        epsilon = max(.02, .6 * (1 - ep/episodes)**1.5)

        result = episode(q, n, horizon, initial, other_bet, rng,

                         learn=True, epsilon=epsilon, allow_skip=allow_skip)

        for name in GOALS: history[name].append(result['success'][name])

        walter_top1.append(result['walter_top1'])

    models = {name: Trained(name, goal, q if name == 'Walter' else np.zeros((1,)),

                            _rolling(history[name]), allow_skip)

              for name, goal in GOALS.items()}

    models['Walter'].reward_curve = _rolling(walter_top1)

    return models



def evaluate(models, n=100, horizon=5000, initial=100., other_bet=.2,

             seed=123, trials=1000, record_paths=25):

    """Counterfactual Walter policies; Gustavo and Jesse stay in EVERY economy.



    Every policy column is a separate shared-economy run, not a separate

    character-training run. Results for Gustavo/Jesse are conditional on Walter's policy.

    """

    n, horizon, trials = int(n), int(horizon), int(trials)

    if trials < 1: raise ValueError('trials must be positive')

    q = models['Walter'].q

    results = {name: {'paths': [], 'bets': [], 'trials': trials} for name in GOALS}

    for pidx, policy in enumerate(POLICIES):

        rng = np.random.default_rng(int(seed) + 1000*pidx)

        observed = {name: [] for name in GOALS}

        wealths = {name: [] for name in GOALS}

        ranks = {name: [] for name in GOALS}

        shares = {name: [] for name in GOALS}

        firsts = {name: [] for name in GOALS}

        concentration = []

        for trial in range(trials):

            record = policy == 'rl' and trial < record_paths

            out = episode(q, n, horizon, initial, other_bet, rng,

                          walter_policy=policy, allow_skip=models['Walter'].allow_skip,

                          record=record)

            w = out['wealth']

            concentration.append([out['gini'], out['top1_share'], out['top5_share'], out['top10_share']])

            for idx, name in enumerate(GOALS):

                observed[name].append(out['success'][name])

                wealths[name].append(float(w[idx]))

                shares[name].append(float(w[idx]/w.sum()))

                ranks[name].append(1 + int(np.count_nonzero(np.delete(w, idx) >= w[idx])))

                firsts[name].append(int(w[idx] > np.max(np.delete(w, idx))))

                if record:

                    results[name]['paths'].append(out['paths'][idx])

                    results[name]['bets'].append(np.asarray(out['bets'][idx]))

        for name in GOALS:

            r = results[name]

            r[policy] = float(np.mean(observed[name]))

            r[policy+'_successes'] = int(np.sum(observed[name]))

            r[policy+'_mean_share'] = float(np.mean(shares[name]))

            r[policy+'_median_share'] = float(np.median(shares[name]))

            r[policy+'_mean_rank'] = float(np.mean(ranks[name]))

            r[policy+'_strict_first'] = float(np.mean(firsts[name]))

            r[policy+'_median_wealth'] = float(np.median(wealths[name]))

            r[policy+'_mean_gini'] = float(np.mean(np.asarray(concentration)[:, 0]))

            r[policy+'_top1_share'] = float(np.mean(np.asarray(concentration)[:, 1]))

            r[policy+'_top5_share'] = float(np.mean(np.asarray(concentration)[:, 2]))

            r[policy+'_top10_share'] = float(np.mean(np.asarray(concentration)[:, 3]))

    for name in GOALS:

        results[name]['paths'] = np.asarray(results[name]['paths'])

    return results





def simulate_condensation(

    models,

    n=100,

    horizon=100_000,

    initial=100.0,

    other_bet=0.20,

    seed=42,

    snapshot_every=1000,

    walter_policy="rl",

 ):

    """

    Run one shared economy and record wealth condensation.



    Agent 0: Walter (learned or baseline policy)

    Agent 1: Gustavo (fixed wager)

    Agent 2: Jesse (random wager)

    Agents 3+: fixed-wager background agents



    No Q-learning updates occur during this simulation.

    """

    n = int(n)

    horizon = int(horizon)

    snapshot_every = int(snapshot_every)



    if n < 3 or horizon < 1 or snapshot_every < 1:

        raise ValueError("Invalid simulation settings")

    if initial <= 0 or not 0 <= other_bet <= 1:

        raise ValueError("Invalid wealth or wager")

    if walter_policy not in POLICIES:

        raise ValueError("Unknown Walter policy")



    rng = np.random.default_rng(int(seed))

    wealth = np.full(n, float(initial))

    q = models["Walter"].q

    allow_skip = models["Walter"].allow_skip



    snapshots = []

    trade_numbers = []

    gini_history = []

    top1_history = []

    top5_history = []

    top10_history = []

    character_history = []



    def record_snapshot(trade_number):

        snapshots.append(wealth.copy())

        trade_numbers.append(trade_number)



        gini_history.append(gini(wealth))

        top1_history.append(top_share(wealth, 0.01))

        top5_history.append(top_share(wealth, 0.05))

        top10_history.append(top_share(wealth, 0.10))



        character_history.append(wealth[:3].copy())



    # Record the initial, perfectly equal economy.

    record_snapshot(0)



    for t in range(horizon):

        i = int(rng.integers(n))

        j = int(rng.integers(n - 1))



        if j >= i:

            j += 1



        if i == 0 or j == 0:

            opponent = j if i == 0 else i



            if walter_policy == "rl":

                state = state_of(wealth, t, horizon, opponent)

                action = choose_action(

                    q, state, rng,

                    epsilon=0.0,

                    allow_skip=allow_skip

                )

                fraction = float(ACTIONS[action])



            elif walter_policy == "baseline":

                fraction = float(other_bet)



            elif walter_policy == "random":

                first = 0 if allow_skip else 1

                action = int(rng.integers(first, len(ACTIONS)))

                fraction = float(ACTIONS[action])



            else:  # aggressive

                fraction = 1.0



        elif i == 2 or j == 2:

            first = 0 if allow_skip else 1

            action = int(rng.integers(first, len(ACTIONS)))

            fraction = float(ACTIONS[action])



        else:

            fraction = float(other_bet)



        # Existing fair Yard Sale trade.

        trade(wealth, i, j, fraction, rng)



        if ((t + 1) % snapshot_every == 0

                or t + 1 == horizon):

            record_snapshot(t + 1)



    return {

        "snapshots": np.asarray(snapshots),

        "trades": np.asarray(trade_numbers),

        "gini": np.asarray(gini_history),

        "top1_share": np.asarray(top1_history),

        "top5_share": np.asarray(top5_history),

        "top10_share": np.asarray(top10_history),

        "characters": np.asarray(character_history),

        "final_wealth": wealth.copy(),

    }






# ---------------------------------------------------------------------------
# Three-character showdown. Independent of the 100-agent RL experiment above.
# ---------------------------------------------------------------------------
SHOWDOWN_NAMES = ('Walter', 'Gustavo', 'Jesse')
SHOWDOWN_BINS = np.array([.10, .25, .50, .75, 1., 1.5, 2., 3., 5., 10.])
SHOWDOWN_TIME_BINS = np.array([.25, .50, .75, .90])
SHOWDOWN_Q_SHAPE = (len(SHOWDOWN_BINS) + 1, len(SHOWDOWN_BINS) + 1,
                    3, len(SHOWDOWN_TIME_BINS) + 1, len(ACTIONS))


def showdown_state(wealth, t, horizon, opponent):
    """Walter's relative wealth versus Gustavo/Jesse, opponent, and progress."""
    return (
        int(np.digitize(wealth[0] / max(wealth[1], 1e-12), SHOWDOWN_BINS)),
        int(np.digitize(wealth[0] / max(wealth[2], 1e-12), SHOWDOWN_BINS)),
        int(opponent),
        int(np.digitize(t / max(horizon, 1), SHOWDOWN_TIME_BINS)),
    )


def showdown_episode(q, horizon=300, initial=100., fixed_bet=.20,
                     seed=42, snapshot_every=5, walter_policy='rl',
                     allow_skip=True, epsilon=0., learn=False, alpha=.08):
    """Three agents, symmetric fair trades, and an optional terminal MC update.

    Each selected agent proposes a wager fraction; the stake is the average
    proposal multiplied by the poorer agent's wealth. A Walter proposal of 0%
    does not veto the other agent's proposal. Each trade conserves wealth.
    """
    horizon, snapshot_every = int(horizon), int(snapshot_every)
    if horizon < 1 or snapshot_every < 1 or initial <= 0 or not 0 <= fixed_bet <= 1:
        raise ValueError('Invalid showdown settings')
    if walter_policy not in POLICIES:
        raise ValueError('Unknown Walter policy')
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(int(seed))
    wealth = np.full(3, float(initial))
    snapshots = [wealth.copy()]
    trade_numbers = [0]
    decisions = []
    wagers = [[], [], []]
    for t in range(horizon):
        i = int(rng.integers(3))
        j = int(rng.integers(2))
        if j >= i:
            j += 1
        proposals = {}
        for k in (i, j):
            if k == 0:
                if walter_policy == 'rl':
                    opponent = j if i == 0 else i
                    state = showdown_state(wealth, t, horizon, opponent)
                    action = choose_action(q, state, rng, epsilon=epsilon,
                                           allow_skip=allow_skip)
                    proposed = float(ACTIONS[action])
                    if learn:
                        decisions.append((state, action))
                elif walter_policy == 'baseline':
                    proposed = float(fixed_bet)
                elif walter_policy == 'random':
                    first = 0 if allow_skip else 1
                    proposed = float(ACTIONS[int(rng.integers(first, len(ACTIONS)))])
                else:
                    proposed = 1.
            elif k == 1:
                proposed = float(fixed_bet)
            else:
                first = 0 if allow_skip else 1
                proposed = float(ACTIONS[int(rng.integers(first, len(ACTIONS)))])
            proposals[k] = proposed
            wagers[k].append(proposed)
        fraction = (proposals[i] + proposals[j]) / 2.
        trade(wealth, i, j, fraction, rng)
        if (t + 1) % snapshot_every == 0 or t + 1 == horizon:
            snapshots.append(wealth.copy())
            trade_numbers.append(t + 1)

    winner = int(np.argmax(wealth)) if np.count_nonzero(
        np.isclose(wealth, wealth.max(), rtol=0, atol=1e-9)) == 1 else -1
    walter_win = int(winner == 0)
    if learn and decisions:
        # Episodic Monte Carlo control: each Walter decision receives the
        # same terminal reward. Reward 1 only for a strict first-place finish.
        # This is intentionally simple; training success is not guaranteed.
        for state, action in decisions:
            key = (*state, action)
            q[key] += alpha * (walter_win - q[key])
    return {
        'snapshots': np.asarray(snapshots),
        'trades': np.asarray(trade_numbers),
        'characters': np.asarray(snapshots),
        'final_wealth': wealth.copy(),
        'winner': SHOWDOWN_NAMES[winner] if winner >= 0 else 'Tie',
        'walter_win': walter_win,
        'wagers': wagers,
        'gini': np.array([gini(s) for s in snapshots]),
    }


def train_showdown(episodes=2000, horizon=300, initial=100., fixed_bet=.20,
                   seed=42, allow_skip=True, alpha=.08):
    """Train a NEW Walter Q-table for the three-person, first-place goal."""
    episodes = int(episodes)
    if episodes < 1:
        raise ValueError('episodes must be positive')
    rng = np.random.default_rng(int(seed))
    q = np.zeros(SHOWDOWN_Q_SHAPE, dtype=float)
    wins = []
    for ep in range(episodes):
        epsilon = max(.05, .6 * (1 - ep / episodes))
        result = showdown_episode(q, horizon, initial, fixed_bet, rng,
                                  snapshot_every=horizon, allow_skip=allow_skip,
                                  epsilon=epsilon, learn=True, alpha=alpha)
        wins.append(result['walter_win'])
    return {
        'q': q,
        'allow_skip': bool(allow_skip),
        'training_win_rate': _rolling(wins),
        'horizon': int(horizon),
        'initial': float(initial),
        'fixed_bet': float(fixed_bet),
    }


def simulate_showdown(model=None, horizon=300, initial=100., fixed_bet=.20,
                      seed=123, snapshot_every=5, walter_policy='rl'):
    """Record a three-bar showdown; no training occurs during playback."""
    if walter_policy == 'rl' and model is None:
        raise ValueError('Train Walter with train_showdown() first, or choose baseline')
    q = model['q'] if model is not None else np.zeros(SHOWDOWN_Q_SHAPE)
    allow_skip = model['allow_skip'] if model is not None else True
    return showdown_episode(q, horizon, initial, fixed_bet, seed,
                            snapshot_every, walter_policy,
                            allow_skip=allow_skip)


def evaluate_showdown(model, trials=500, horizon=None, initial=None,
                      fixed_bet=None, seed=123):
    """Compare strict first-place rates across Walter policies."""
    trials = int(trials)
    if trials < 1:
        raise ValueError('trials must be positive')
    horizon = model['horizon'] if horizon is None else int(horizon)
    initial = model['initial'] if initial is None else float(initial)
    fixed_bet = model['fixed_bet'] if fixed_bet is None else float(fixed_bet)
    output = {}
    for policy in POLICIES:
        wins = np.zeros(3, dtype=int)
        ties = 0
        for trial in range(trials):
            result = simulate_showdown(model, horizon, initial, fixed_bet,
                                       seed + trial, horizon, policy)
            if result['winner'] == 'Tie':
                ties += 1
            else:
                wins[SHOWDOWN_NAMES.index(result['winner'])] += 1
        output[policy] = {
            **{name: wins[idx] / trials for idx, name in enumerate(SHOWDOWN_NAMES)},
            'tie': ties / trials,
            'trials': trials,
        }
    return output


if __name__ == "__main__":



    # 1. Train Walter in the shared economy

    print("\nTraining Walter's Q-learning agent...")



    models = train_all(

        n=100,

        horizon=10_000,

        episodes=1_000,

        initial=100.0,

        other_bet=0.20,

        seed=42,

        allow_skip=True

    )



    # 2. Evaluate Walter, Gustavo and Jesse

    print("\nEvaluating all three characters...")



    results = evaluate(

        models,

        n=100,

        horizon=10_000,

        initial=100.0,

        other_bet=0.20,

        seed=123,

        trials=200

    )



    for name, r in results.items():

        print(f"\n{name}")

        print(f"  Goal success: {r['rl']:.1%}")

        print(f"  Median wealth: ${r['rl_median_wealth']:.2f}")

        print(f"  Mean Gini: {r['rl_mean_gini']:.3f}")



    # 3. Run long-term wealth condensation

    print("\nSimulating 100,000 fair trades...")



    simulation = simulate_condensation(

        models,

        n=100,

        horizon=100_000,

        initial=100.0,

        other_bet=0.20,

        seed=42,

        snapshot_every=1000,

        walter_policy="rl"

    )



    # 4. Display final inequality statistics

    print("\nWEALTH CONDENSATION RESULTS")

    print(f"Final Gini: {simulation['gini'][-1]:.3f}")

    print(f"Richest 1% share: {simulation['top1_share'][-1]:.1%}")

    print(f"Richest 5% share: {simulation['top5_share'][-1]:.1%}")

    print(f"Richest 10% share: {simulation['top10_share'][-1]:.1%}")



    # 5. Display character fortunes

    print("\nFINAL CHARACTER WEALTH")



    for name, wealth in zip(

        ("Walter", "Gustavo", "Jesse"),

        simulation["characters"][-1]

    ):

        print(f"{name}: ${wealth:.2f}")



    # 6. Save snapshots for visualization

    np.savez_compressed(

        "wealth_condensation_snapshots.npz",

        **simulation

    )



    print("\nSnapshots saved successfully.")

    print(f"Total snapshots: {len(simulation['trades'])}")