# Yard Sale Model · Yard Sale Model

An independent computational economics project exploring wealth condensation, stochastic trade, and reinforcement learning. Visual storytelling applications are optional.

![Dashboard concept](assets/dashboard_concept.png)

> The concept image is an **artistic mockup**, not measured experimental output.
> Run the app to generate the actual plots and success rates.

## Run locally

From this directory (`AI_Educate/economics/yard_sale`):

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
python app.py
```

Visit **http://127.0.0.1:7860**. On Windows activate with `.venv\\Scripts\\activate`.

## The two tabs

**1. Wealth Condensation:** all agents start equally wealthy. Two random agents trade; the stake is a fraction of the poorer agent's wealth, and each wins with probability 1/2. Plot Gini, richest share, ranked distribution and Lorenz curve; scrub through saved snapshots.

**2. Can a Genius Beat It?:** three separate tabular Monte Carlo RL learners optimize different *terminal binary rewards*: top 1%, above the median, and avoiding the bottom 10%. Each learner chooses a wager from 0%, 2%, 5%, 10%, 20%, 40%, 80%, 100% of the poorer party's wealth. The 0% action can be disabled. Other agents wager a fixed 20% by default. An independent evaluation compares learned policy with a non-learning 20% bettor. Graphs show rolling training success, evaluation wealth paths, learned policy and success-rate comparisons.

### Important experiment design notes

- **No invented findings:** values on the dashboard are computed from simulation runs. The mockup is illustrative only.
- **Population matters:** for 100 agents, top 1% means first place. With fewer than 100, the top-1% threshold is rounded up to one agent.
- **Learning is not guaranteed:** terminal-reward RL can be noisy and slow; policy outcomes depend on seed, training budget, population and horizon. This is a simple *tabular Monte Carlo control* prototype, not a neural network or a claim of optimal strategy.
- **Baseline definition:** the benchmark is a *fixed-stake 20% player*, not a player that samples wagers uniformly. The benchmark wager follows the "other agents" control.
- **Comparison caveat:** baseline and learner have independently sampled episodes; for stronger statistical claims, use more evaluation trials, multiple training seeds and confidence intervals.
- **Environment convention:** every step gives the learner one trade with a random other agent **and** executes one trade between non-learning agents. This is a simplified experimental economy; compare results only under the same convention.
- **Finite precision:** wealth may become very small, and exact zero may be approached rather than reached.
- **Scientific scope:** fair closed-system transactions only; not a forecast for actual economies.

## Test

```bash
pytest -q
```

## Files

```text
app.py                         # Gradio, exactly two tabs
yard_sale_model/economy.py     # Fair transactions, Gini, snapshots
yard_sale_model/rl.py          # Learner, rewards, training, evaluation
yard_sale_model/plots.py       # Plotly dashboard figures
tests/test_models.py           # Conservation and smoke tests
assets/                        # Concept illustrations
```

## Creative framing

Walter, Jesse and Gustavo illustrate how **fair transactions do not guarantee fair final distributions**. The RL episode asks whether different definitions of success produce different learned betting behavior. Use real evaluated numbers when recording narration, not the placeholders or values in concept art.
