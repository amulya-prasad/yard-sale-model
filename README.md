# The Yard Sale Model

### How Fair Trades Can Create Extreme Wealth Inequality

An interactive computational economics project exploring **wealth condensation, stochastic trading, and reinforcement learning** through a fictional competition between Walter, Gustavo, and Jesse.

What happens when everyone starts equally wealthy, every transaction has a 50/50 outcome, and nobody can predict the next winner? Can changing a player's betting strategy change the *probability* of reaching a particular goal, even when every trade is fair?

Built with **Python, NumPy, Plotly, and Gradio**.

> **Fair transactions do not necessarily produce equal distributions of wealth.** A fair coin flip guarantees neither an equal outcome nor an equal final ranking.

---

## The Big Picture

![Yard Sale Model Poster](assets/yard_sale_poster.png)

The project has two related parts:

1. **A conventional Yard Sale economy**, where randomly paired agents exchange a fraction of the poorer participant's wealth.
2. **Character-based strategy experiments**, where Walter learns wager choices, Gustavo uses a fixed wager, and Jesse randomizes his wager.

All simulations use a closed economy: money is transferred between participants rather than created or destroyed.

---

## 1. The Yard Sale Model

![Fair Trades Illustration](assets/fair_trades_illustration.png)

Two agents, with wealth $w_i$ and $w_j$, trade a stake equal to a fraction of the poorer agent's wealth:

$$
\Delta w = \beta\min(w_i,w_j)
$$

The outcome is determined by a fair coin flip. If agent $i$ wins:

$$
w_i' = w_i + \Delta w, \qquad w_j' = w_j - \Delta w
$$

If agent $j$ wins, the signs are reversed. Each participant has a 50% chance of winning, and the total wealth remains constant:

$$
\sum_{i=1}^{N}w_i(t) = \sum_{i=1}^{N}w_i(0)
$$

**Example:** Walter and Jesse each begin with 100 units. At a 20% wager, 20 units change hands. If Walter wins, their wealth becomes 120 and 80. On the next trade, the same 20% rule puts only $0.2\times80=16$ units at stake. The poorer player's wealth determines the size of the transfer.

The model isolates the effects of repeated random exchange; it does not model salaries, production, taxation, credit, or differences in skill.

---

## 2. Experiment One: Wealth Condensation

![Stochastic Simulation](assets/stochastic_sim.png)

The baseline simulation in `src/economy.py` begins with equal wealth and executes a configurable number of random trades. For each trade, it samples **two distinct agents**, applies the same fixed wager fraction, and chooses the winner with equal probability.

The simulation records snapshots of the wealth distribution and tracks:

- **Gini coefficient:** a summary measure of wealth inequality, ranging from 0 for equality toward 1 for extreme concentration.
- **Richest-agent wealth share:** the fraction of total wealth held by the single richest participant.
- **Wealth distribution snapshots:** how individual fortunes change across the simulation.
- **Lorenz curve and related plots:** visual summaries of inequality in the Gradio dashboard.

The baseline simulator checks that total wealth is conserved at the end of the run. It uses **one randomly selected pair per trade**, not a simultaneous pairing of every participant each round.

---

## 3. Experiment Two: Walter, Gustavo, and Jesse in One Economy

![Reinforcement Learning Agents](assets/rl_agents.png)

The larger-population strategy experiment is implemented in **`src/rl_agents_new_strategy.py`**. All three named characters coexist with additional background agents in the *same simulated town*.

| Character | Strategy | Reported goal |
|---|---|---|
| **Walter — Concentrator** | Learns wager choices using a tabular Q-value policy | Hold at least **25% of the town's total wealth** |
| **Gustavo — Capitalist** | Uses the fixed wager (20% by default) | Finish in the **top 10%** by rank |
| **Jesse — Grower** | Selects a wager fraction randomly | Finish with at least **150% of starting wealth** |

**Only Walter learns.** Gustavo and Jesse do not have separately trained Q-tables. Their goal outcomes are measured after each shared-economy episode.

### Walter's training target versus the displayed diagnostics

There are **two different success measures** in the implementation:

- Walter's **training success signal** is a **top-1% finish** (for 100 agents, a strict first-place finish).
- Walter's named **Concentrator diagnostic** checks whether he owns **at least 25% of all wealth**.

These are not equivalent. A player can finish first without holding a quarter of the economy. The README and plots should not treat the 25% threshold as the terminal reward used to train Walter.

### What Walter observes

Before a trade involving Walter, the code discretizes four observations:

1. **Relative rank:** the fraction of other participants richer than Walter.
2. **Relative wealth:** Walter's wealth divided by the town's mean wealth.
3. **Opponent wealth ratio:** the selected opponent's wealth relative to Walter's.
4. **Episode progress:** how far the simulation has advanced.

These observations index a tabular Q-value array. Walter selects a wager using an exploration-versus-exploitation policy: sometimes he tries a random permitted wager, and otherwise he chooses an action with a high estimated Q-value.

The available wager fractions are:

`0%, 2%, 5%, 10%, 20%, 40%, 80%, 100%`

The **0% action is optional** and can be disabled in the dashboard.

### How the larger-economy learning works

Walter's learning update in this experiment is **not pure terminal-only Monte Carlo control**. It uses a tabular **bootstrapped Q-value update with a shaped intermediate signal**, based on changes in a bounded wealth-related potential. A terminal top-1% success indicator is included at the final update.

That distinction is important: the shaping signal influences what Walter learns, so the experiment should not be described as training exclusively from a final binary reward.

### Trading convention

One randomly selected pair trades per step. Walter chooses the stake fraction **when he participates**; Jesse randomizes the stake fraction **when he participates in a trade that does not involve Walter**; other trades use the fixed background fraction. In a Walter–Jesse trade, Walter's selected fraction takes precedence.

Consequently, participants are **not guaranteed to trade once per step**, and Walter does not control every trade in the economy.

### Evaluation and baselines

The evaluation reruns the shared economy under four alternative **Walter policies**:

| Walter policy | Behavior |
|---|---|
| `rl` | Uses Walter's learned Q-table |
| `baseline` | Uses the fixed background wager |
| `random` | Chooses a random permitted wager |
| `aggressive` | Always wagers 100% |

Gustavo and Jesse remain present in **every** evaluation run. Their measured outcomes can change when Walter's policy changes, even though Gustavo and Jesse do not learn.

Reported metrics include goal-success rates, median wealth, average rank, strict-first-place frequency, Gini coefficient, and wealth shares held by the richest 1%, 5%, and 10% of agents. The dashboard also plots training behavior, wealth trajectories, and betting policies.

Evaluation uses new seeded simulations, with separate random streams for Walter's policy comparisons. **Success in one run does not establish that a strategy consistently outperforms the alternatives.**

---

## 4. Experiment Three: Watch Wealth Condense

The third Gradio tab animates a **single shared economy** over many random trades. Walter, Gustavo, and Jesse appear alongside the background population.

Choose Walter's strategy (`rl`, `baseline`, `random`, or `aggressive`), run the simulation, then play, pause, or scrub through recorded snapshots.

The animation records:

- Gini coefficient over time.
- Wealth shares of the richest 1%, 5%, and 10%.
- Wealth trajectories of Walter, Gustavo, and Jesse.
- Full-population wealth snapshots for distribution and Lorenz plots.

This is a **simulation/playback mode**: it does not update Walter's Q-table during playback. To use the learned policy, first train Walter in the second tab. A policy trained with different settings may not generalize reliably.

---

## 5. Experiment Four: The Bread Wars — Three-Character Showdown

The fourth tab is a **separate three-person experiment**, with **no background agents**. Walter, Gustavo, and Jesse begin with identical wealth and repeatedly trade in randomly selected pairs.

| Character | Showdown behavior |
|---|---|
| **Walter** | Learns wager proposals intended to maximize strict first-place finishes |
| **Gustavo** | Proposes the same fixed wager fraction |
| **Jesse** | Proposes a randomly selected wager fraction |

### How a showdown trade differs

In this mode, **both selected participants propose a wager fraction**. The stake fraction is their average:

$$
\beta_{\mathrm{trade}} = \frac{\beta_i+\beta_j}{2}
$$

The amount transferred is still determined by the poorer participant's wealth:

$$
\Delta w=\beta_{\mathrm{trade}}\min(w_i,w_j)
$$

For example, if Walter proposes 80% and Gustavo proposes 20%, the trade uses a **50%** wager fraction. A 0% proposal from Walter does **not** cancel the other participant's wager.

### How Walter learns in the showdown

This is the part of the project that uses **episodic terminal-reward Monte Carlo-style updates**. Walter's showdown Q-table is separate from his larger-economy Q-table. The observations describe his wealth relative to Gustavo and Jesse, the opponent, and the progress of the showdown.

After each showdown, Walter receives **1 if he finishes strictly first**, or **0 otherwise**. His visited state-action values are updated using that final result. Gustavo and Jesse remain non-learning competitors.

The dashboard compares Walter's learned policy with fixed-wager, random-wager, and always-100% policies. It displays training win rates, strict-first-place results for all three characters, animated wealth bars, and wealth trajectories.

**Important:** The three-person showdown and the larger-population experiment use different wager-setting conventions and different learning updates. Their success rates should not be compared as if they came from identical environments.

---

## 6. Interactive Dashboard

Launch `app.py` to open four Gradio tabs:

| Tab | What you can do |
|---|---|
| **1 · Wealth Condensation** | Run the baseline stochastic model and explore inequality snapshots |
| **2 · Walter vs Gustavo vs Jesse · RL** | Train Walter in a shared town and compare strategy outcomes |
| **3 · Watch Wealth Condense** | Animate wealth concentration and character trajectories |
| **4 · Walter vs Gustavo vs Jesse · Showdown** | Train Walter for a three-person first-place contest and animate the competition |

Plots and statistics are computed from simulations rather than fabricated examples. The README does **not** claim that PPO is connected to this dashboard: the provided `app.py` imports the tabular implementation in `src/rl_agents_new_strategy.py`.

---

## 7. Run Locally

### Clone the repository

```bash
git clone https://github.com/amulya-prasad/yard-sale-model.git
cd yard-sale-model
```

### Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

The application requires NumPy, Gradio, and Plotly. If your requirements file is missing one of these packages, add it before running.

### Start the dashboard

```bash
python app.py
```

Open **http://127.0.0.1:7860**.

### Run the strategy module directly

The strategy file also includes a command-line demonstration that trains Walter, evaluates the shared economy, runs a longer condensation simulation, and saves a compressed NumPy snapshot archive:

```bash
python -m src.rl_agents_new_strategy
```

This can be computationally intensive with the configured training and simulation lengths. The direct-run example saves `wealth_condensation_snapshots.npz` in the current working directory.

---

## 8. Project Structure

```text
yard-sale-model/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── economy.py
│   ├── plots.py
│   └── rl_agents_new_strategy.py
├── assets/
│   ├── yard_sale_poster.png
│   ├── fair_trades_illustration.png
│   ├── stochastic_sim.png
│   └── rl_agents.png
└── outputs/                    # optional location for exported results
```

The tree lists the **files required or referenced by the provided current scripts**, not an independently verified inventory of your GitHub repository. Your local repository may also contain earlier experiments, test files, or PPO prototypes.

| File | Purpose |
|---|---|
| `app.py` | Four-tab Gradio interface |
| `src/economy.py` | Baseline fair-trade simulation, Gini, and wealth conservation |
| `src/rl_agents_new_strategy.py` | Shared-economy learning, policy evaluation, condensation playback, and separate three-person showdown |
| `src/plots.py` | Plotly chart functions used by the dashboard |
| `assets/` | Illustrations and experiment imagery |
| `outputs/` | Optional directory for future exported results |

---

## 9. Assumptions and Limitations

- **Stylized closed economy:** No production, income, debt, taxes, or redistribution.
- **Fair trade outcomes:** Each trade winner is chosen with equal probability; betting policies influence the *size* of transfers, not the coin flip.
- **Random pairing:** Each step samples one pair; there is no guarantee that every baker trades equally often over a finite run.
- **Different learning methods:** The larger shared-economy experiment uses bootstrapped Q-value updates with a shaping signal; the three-person showdown uses terminal-reward Monte Carlo-style updates.
- **Only Walter learns:** Gustavo and Jesse follow fixed or random strategies, respectively.
- **Different trading conventions:** The shared economy uses one applicable wager fraction per trade; the showdown averages both participants' proposals.
- **Optional zero wager:** In the shared economy, a 0% wager can prevent a wealth transfer when Walter controls the selected trade; in the showdown, a 0% *proposal* does not necessarily make the stake zero.
- **Finite-run variability:** A strategy can appear successful due to luck. Robust comparisons require independent training seeds, many evaluation episodes, and uncertainty intervals.
- **Numerical behavior:** Very small wealth values and ties can affect rankings and terminal goal definitions.
- **Not a forecast:** The simulations demonstrate properties of simplified models, not predictions about real financial markets.

---

## 10. Future Directions

- Evaluate training stability over multiple random seeds and report confidence intervals.
- Export reproducible CSV datasets for figures and the accompanying video essay.
- Compare tabular learning with **PPO** in a carefully matched environment.
- Investigate policy behavior by rank, opponent wealth, and time remaining.
- Extend the analysis to an affine wealth model with redistribution, using explicitly documented assumptions.

---

## Author

**Amulya Prasad**

Independent computational experiments in economics, mathematics, and physics using AI.

*An exploration of how randomness, incentives, and repeated interactions can shape the distribution of wealth.*