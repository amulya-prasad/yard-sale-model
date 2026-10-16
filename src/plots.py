"""Reusable parchment-themed Plotly figures for the shared Yard Sale economy.



Supports:

- Original wealth-condensation tab (`condensation_charts`)

- Shared-economy RL tab (`rl_charts`)

- Interactive snapshot playback (`condensation_animation_charts`)

"""



import numpy as np

import plotly.graph_objects as go

from plotly.subplots import make_subplots



PAPER = "#F7F4EA"

INK = "#292620"

GREEN = "#6B927C"

ORANGE = "#C47B58"

BLUE = "#6588AA"

PURPLE = "#9285AD"

GRAY = "#A4A09A"

PALETTE = {

    "Walter": ORANGE, "Gustavo": GREEN, "Jesse": PURPLE,

    "Gambler": ORANGE, "Comfortable": BLUE, "Survivor": GREEN,

}

CHARACTERS = ("Walter", "Gustavo", "Jesse")

POLICY_NAMES = {"rl": "Walter RL", "baseline": "Fixed wager",

                "random": "Random wager", "aggressive": "100% wager"}





def style(fig, title, height=350):

    fig.update_layout(

        title=dict(text=title, x=0.02),

        paper_bgcolor=PAPER, plot_bgcolor=PAPER,

        font=dict(color=INK, family="Arial"), height=height,

        margin=dict(l=55, r=25, t=75, b=55),

        legend=dict(orientation="h", y=1.14, x=0),

        hovermode="closest",

    )

    fig.update_xaxes(showgrid=False, linecolor=INK)

    fig.update_yaxes(gridcolor="rgba(41,38,32,.12)", zeroline=False)

    return fig





def _gini(wealth):

    x = np.sort(np.asarray(wealth, dtype=float))

    total = x.sum()

    if len(x) == 0 or total <= 0:

        return 0.0

    n = len(x)

    return float(2 * np.dot(np.arange(1, n + 1), x) / (n * total) - (n + 1) / n)





def _top_share(wealth, fraction):

    x = np.asarray(wealth, dtype=float)

    if len(x) == 0 or x.sum() <= 0:

        return 0.0

    k = max(1, int(np.ceil(fraction * len(x))))

    return float(np.sort(x)[-k:].sum() / x.sum())





def condensation_charts(history, snapshots, index):

    """Preserve the original condensation-tab API.



    history columns: trade count, Gini, richest-agent wealth share.

    """

    history = np.asarray(history, dtype=float)

    snapshots = np.asarray(snapshots, dtype=float)

    index = int(np.clip(index, 0, len(snapshots) - 1))

    wealth = snapshots[index]

    tick = int(history[index, 0])



    timeline = go.Figure()

    timeline.add_scatter(x=history[:, 0], y=history[:, 1], name="Gini",

                         line=dict(color=ORANGE, width=3))

    timeline.add_scatter(x=history[:, 0], y=history[:, 2], name="Richest share",

                         line=dict(color=GREEN, width=3))

    timeline.add_vline(x=tick, line_dash="dot", line_color=INK)

    timeline.update_yaxes(range=[0, 1])

    timeline.update_xaxes(title="Trades")

    style(timeline, "Inequality over time")



    distribution = go.Figure(go.Bar(

        x=np.arange(1, len(wealth) + 1), y=np.sort(wealth)[::-1],

        marker_color=[ORANGE] + [GREEN] * (len(wealth) - 1),

    ))

    distribution.update_xaxes(title="Wealth rank (richest first)")

    distribution.update_yaxes(title="Wealth")

    style(distribution, f"Wealth distribution · trade {tick:,}")



    sorted_w = np.sort(wealth)

    total = sorted_w.sum()

    lorenz_y = np.r_[0, np.cumsum(sorted_w) / total] if total > 0 else np.zeros(len(wealth) + 1)

    lorenz = go.Figure()

    lorenz.add_scatter(x=[0, 1], y=[0, 1], name="Perfect equality",

                       line=dict(color=INK, dash="dash"))

    lorenz.add_scatter(x=np.linspace(0, 1, len(wealth) + 1), y=lorenz_y,

                       name="Observed", line=dict(color=ORANGE, width=3))

    lorenz.update_xaxes(title="Cumulative population share", range=[0, 1])

    lorenz.update_yaxes(title="Cumulative wealth share", range=[0, 1])

    style(lorenz, "Lorenz curve")



    stats = (f"### Snapshot · {tick:,} trades\n"

             f"**Gini:** {history[index, 1]:.3f}  |  "

             f"**Richest agent:** {_top_share(wealth, 1 / len(wealth)):.1%}  |  "

             f"**Top 10%:** {_top_share(wealth, .10):.1%}  |  "

             f"**Total wealth:** {wealth.sum():,.0f}")

    return timeline, distribution, lorenz, stats





def condensation_animation_charts(simulation, index):

    """Render one interactive frame from simulate_condensation() output.



    Returns (inequality_timeline, ranked_distribution, character_paths,

             lorenz_curve, markdown_stats). Call on Gradio slider changes.

    """

    trades = np.asarray(simulation["trades"])

    snapshots = np.asarray(simulation["snapshots"], dtype=float)

    characters = np.asarray(simulation["characters"], dtype=float)

    if snapshots.ndim != 2 or len(trades) != len(snapshots):

        raise ValueError("Expected snapshots of shape (frames, agents) and matching trades")

    if characters.shape != (len(trades), 3):

        raise ValueError("Expected character wealth history of shape (frames, 3)")

    index = int(np.clip(index, 0, len(trades) - 1))

    tick = int(trades[index])

    wealth = snapshots[index]

    n = len(wealth)



    timeline = go.Figure()

    for key, label, color in (

        ("gini", "Gini", ORANGE),

        ("top1_share", "Top 1%", BLUE),

        ("top5_share", "Top 5%", PURPLE),

        ("top10_share", "Top 10%", GREEN),

    ):

        timeline.add_scatter(x=trades, y=simulation[key], name=label,

                             mode="lines", line=dict(color=color, width=2.5))

    timeline.add_vline(x=tick, line_color=INK, line_dash="dot")

    timeline.update_xaxes(title="Number of trades")

    timeline.update_yaxes(title="Gini / fraction of total wealth", range=[0, 1])

    style(timeline, "How inequality develops", height=390)



    # Sort by wealth while retaining each agent's identity for highlighting.

    order = np.argsort(-wealth, kind="stable")

    colors = [PALETTE[CHARACTERS[i]] if i < 3 else GRAY for i in order]

    labels = [CHARACTERS[i] if i < 3 else f"Agent {i + 1}" for i in order]

    distribution = go.Figure(go.Bar(

        x=np.arange(1, n + 1), y=wealth[order], marker_color=colors,

        customdata=np.asarray(labels).reshape(-1, 1),

        hovertemplate="Rank %{x}<br>%{customdata[0]}<br>Wealth $%{y:,.2f}<extra></extra>",

    ))

    distribution.update_xaxes(title="Agent rank (richest to poorest)")

    distribution.update_yaxes(title="Current wealth ($)", rangemode="tozero")

    style(distribution, f"Wealth distribution · trade {tick:,}", height=400)



    paths = go.Figure()

    for i, name in enumerate(CHARACTERS):

        paths.add_scatter(x=trades[:index + 1], y=characters[:index + 1, i],

                          name=name, mode="lines+markers",

                          line=dict(color=PALETTE[name], width=3),

                          marker=dict(size=4))

    paths.update_xaxes(title="Number of trades")

    paths.update_yaxes(title="Character wealth ($)", rangemode="tozero")

    style(paths, "Walter, Gustavo and Jesse · same economy", height=350)



    sorted_w = np.sort(wealth)

    total = sorted_w.sum()

    lorenz_y = np.r_[0, np.cumsum(sorted_w) / total] if total > 0 else np.zeros(n + 1)

    lorenz = go.Figure()

    lorenz.add_scatter(x=[0, 1], y=[0, 1], name="Perfect equality",

                       line=dict(color=INK, dash="dash"))

    lorenz.add_scatter(x=np.linspace(0, 1, n + 1), y=lorenz_y,

                       name="Observed", line=dict(color=ORANGE, width=3))

    lorenz.update_xaxes(title="Cumulative population share", range=[0, 1])

    lorenz.update_yaxes(title="Cumulative wealth share", range=[0, 1])

    style(lorenz, "Lorenz curve", height=350)



    stats = (

        f"### Trade {tick:,} · Snapshot {index + 1}/{len(trades)}\n\n"

        f"**Gini:** {float(simulation['gini'][index]):.3f}  |  "

        f"**Richest 1%:** {float(simulation['top1_share'][index]):.1%}  |  "

        f"**Richest 5%:** {float(simulation['top5_share'][index]):.1%}  |  "

        f"**Richest 10%:** {float(simulation['top10_share'][index]):.1%}\n\n"

        f"**Walter:** ${wealth[0]:,.2f}  |  "

        f"**Gustavo:** ${wealth[1]:,.2f}  |  "

        f"**Jesse:** ${wealth[2]:,.2f}  |  "

        f"**Total:** ${wealth.sum():,.2f}"

    )

    return timeline, distribution, paths, lorenz, stats





def rl_charts(models, results, n, horizon, initial):

    """Shared-economy RL plots; returns original five-output signature.



    Only Walter has a learned Q-table. Gustavo and Jesse have fixed/random

    policies, so plotting a learned wager policy for them would be misleading.

    """

    names = [name for name in CHARACTERS if name in models and name in results]

    if not names:

        raise ValueError("Expected Walter, Gustavo and Jesse models/results")



    training = go.Figure()

    for name in names:

        curve = np.asarray(models[name].curve)

        training.add_scatter(x=np.arange(1, len(curve) + 1), y=curve,

                             name=name, line=dict(color=PALETTE[name], width=2))

    walter_reward = getattr(models.get("Walter"), "reward_curve", None)

    if walter_reward is not None:

        reward = np.asarray(walter_reward)

        training.add_scatter(x=np.arange(1, len(reward) + 1), y=reward,

                             name="Walter top 1%", line=dict(color=INK, dash="dash"))

    training.update_yaxes(title="Rolling goal success", range=[0, 1])

    training.update_xaxes(title="Training episode")

    style(training, "Training · shared economy", height=360)



    paths = make_subplots(rows=1, cols=len(names), subplot_titles=names)

    for col, name in enumerate(names, start=1):

        data = np.asarray(results[name].get("paths", []), dtype=float)

        if data.ndim != 2 or data.size == 0:

            continue

        x = np.linspace(0, horizon, data.shape[1])

        for path in data:

            paths.add_scatter(x=x, y=path, mode="lines",

                              line=dict(color="rgba(90,90,90,.18)", width=1),

                              showlegend=False, row=1, col=col)

        paths.add_scatter(x=x, y=np.median(data, axis=0), mode="lines",

                          line=dict(color=PALETTE[name], width=3),

                          name=f"{name} median", showlegend=False,

                          row=1, col=col)

    paths.update_xaxes(title="Trades")

    paths.update_yaxes(title="Wealth ($)", rangemode="tozero")

    style(paths, "Evaluation paths · individual runs and median", height=370)



    policies = go.Figure()

    # Actual wager histories from evaluation, not fake learned Q-tables.

    for name in names:

        bets = results[name].get("bets", [])

        if not bets:

            continue

        arrays = [np.asarray(b, dtype=float) for b in bets if len(b)]

        if not arrays:

            continue

        values = np.concatenate(arrays)

        actions, counts = np.unique(np.round(values * 100, 4), return_counts=True)

        policies.add_bar(x=[f"{a:g}%" for a in actions], y=counts / counts.sum(),

                         name=name, marker_color=PALETTE[name])

    policies.update_layout(barmode="group")

    policies.update_xaxes(title="Wager (% of poorer agent's wealth)")

    policies.update_yaxes(title="Fraction of recorded wagers", tickformat=".0%")

    style(policies, "Wagers used in recorded evaluation episodes", height=370)



    compare = go.Figure()

    for policy, color in (("rl", ORANGE), ("baseline", GREEN),

                          ("random", PURPLE), ("aggressive", GRAY)):

        compare.add_bar(x=names, y=[100 * results[name].get(policy, 0) for name in names],

                        name=POLICY_NAMES[policy], marker_color=color)

    compare.update_layout(barmode="group")

    compare.update_yaxes(title="Goal success rate (%)", range=[0, 100])

    style(compare, "Character goal success by Walter's policy", height=370)



    summary = ("### Shared-economy evaluation\n\n"

               "| Character | RL Walter | Fixed Walter | Random Walter | Aggressive Walter | Trials |\n"

               "|---|---:|---:|---:|---:|---:|\n")

    for name in names:

        r = results[name]

        summary += (f"| {name} | {r.get('rl', 0):.1%} | "

                    f"{r.get('baseline', 0):.1%} | {r.get('random', 0):.1%} | "

                    f"{r.get('aggressive', 0):.1%} | {r.get('trials', 0)} |\n")

    summary += ("\nOnly **Walter** learns. Gustavo always uses the configured fixed wager; "

                "Jesse wagers randomly. Every evaluation run contains all three characters "

                "and the background agents. The columns compare Walter's policy, not "

                "separately trained agents. Goal success is not guaranteed.")

    return training, paths, policies, compare, summary

# ---------------------------------------------------------------------------
# Three-character showdown (independent of the 100-agent economy)
# ---------------------------------------------------------------------------
def showdown_charts(simulation, index):
    """Render a three-bar showdown frame from simulate_showdown().

    Returns (wealth_bars, wealth_paths, markdown_stats) for Gradio plots.
    The simulation contains exactly Walter, Gustavo, and Jesse.
    """
    snapshots = np.asarray(simulation['snapshots'], dtype=float)
    trades = np.asarray(simulation['trades'])
    if snapshots.ndim != 2 or snapshots.shape[1] != 3:
        raise ValueError('Showdown snapshots must have shape (frames, 3)')
    if trades.ndim != 1 or len(trades) != len(snapshots) or not len(trades):
        raise ValueError('Showdown trades must match the number of snapshots')
    if not np.all(np.isfinite(snapshots)) or np.any(snapshots < -1e-8):
        raise ValueError('Showdown wealth must be finite and nonnegative')

    index = int(np.clip(index, 0, len(trades) - 1))
    wealth = snapshots[index]
    tick = int(trades[index])
    total = float(wealth.sum())
    names = list(CHARACTERS)
    colors = [PALETTE[name] for name in names]

    bars = go.Figure(go.Bar(
        x=names, y=wealth, marker_color=colors,
        text=[f'${v:,.2f}' for v in wealth], textposition='outside',
        hovertemplate='%{x}<br>Wealth: $%{y:,.2f}<extra></extra>',
        width=[0.55] * 3,
    ))
    bars.update_xaxes(title=None, categoryorder='array', categoryarray=names)
    bars.update_yaxes(title='Current wealth ($)', range=[0, max(total * 1.12, 1)])
    bars.update_layout(showlegend=False, bargap=0.4)
    style(bars, f'The Bread Wars · trade {tick:,}', height=430)

    paths = go.Figure()
    for i, name in enumerate(names):
        paths.add_scatter(
            x=trades[:index + 1], y=snapshots[:index + 1, i],
            name=name, mode='lines+markers',
            line=dict(color=PALETTE[name], width=3), marker=dict(size=5),
            hovertemplate='Trade %{x:,}<br>Wealth $%{y:,.2f}<extra>' + name + '</extra>',
        )
    paths.update_xaxes(title='Number of trades', range=[0, max(int(trades[-1]), 1)])
    paths.update_yaxes(title='Wealth ($)', range=[0, max(total * 1.05, 1)])
    style(paths, 'Three-character wealth trajectory', height=350)

    order = np.argsort(-wealth, kind='stable')
    ranking = ', '.join(f'{k + 1}. {names[i]} (${wealth[i]:,.2f})'
                        for k, i in enumerate(order))
    current_leaders = [names[i] for i in range(3)
                       if np.isclose(wealth[i], wealth.max(), rtol=0, atol=1e-9)]
    leader = current_leaders[0] if len(current_leaders) == 1 else 'Tie'
    status = ('Final winner' if index == len(trades) - 1 else 'Current leader')
    stats = (
        f'### Trade {tick:,} · Snapshot {index + 1}/{len(trades)}\n\n'
        f'**{status}:** {leader}  |  **Total wealth:** ${total:,.2f}\n\n'
        f'**Ranking:** {ranking}\n\n'
        '**Strategies:** Walter — selected policy; Gustavo — fixed wager; '
        'Jesse — random wager. Every trade has a fair 50/50 outcome.'
    )
    return bars, paths, stats


def showdown_training_chart(model):
    """Plot Walter's rolling strict-first-place rate during showdown training."""
    values = np.asarray(model['training_win_rate'], dtype=float)
    fig = go.Figure()
    fig.add_scatter(x=np.arange(1, len(values) + 1), y=values,
                    name='Walter', line=dict(color=ORANGE, width=3))
    fig.update_xaxes(title='Training episode')
    fig.update_yaxes(title='Rolling first-place rate', range=[0, 1], tickformat='.0%')
    return style(fig, 'Walter learns · three-character showdown', height=340)


def showdown_evaluation_chart(results):
    """Grouped win-rate bars from evaluate_showdown() for all three characters."""
    fig = go.Figure()
    policies = [p for p in POLICY_NAMES if p in results]
    for name in CHARACTERS:
        fig.add_bar(x=[POLICY_NAMES[p] for p in policies],
                    y=[100 * float(results[p][name]) for p in policies],
                    name=name, marker_color=PALETTE[name])
    fig.update_layout(barmode='group')
    fig.update_xaxes(title='Walter strategy')
    fig.update_yaxes(title='Strict first-place finishes (%)', range=[0, 100])
    return style(fig, 'Who wins across repeated showdowns?', height=370)

