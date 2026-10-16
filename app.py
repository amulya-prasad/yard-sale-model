"""Yard Sale Model interactive lab. Run: python app.py"""
import gradio as gr
from src.economy import simulate
from src.rl_agents_new_strategy import (
    train_all, evaluate, simulate_condensation, train_showdown,
    simulate_showdown, evaluate_showdown, Q_SHAPE, Trained,
)
from src.plots import condensation_charts, rl_charts, condensation_animation_charts
import numpy as np
import plotly.graph_objects as go


def run_stochastic(n, initial, wager, trades, seed):
    history, snapshots = simulate(int(n), float(initial), float(wager)/100, int(trades), int(seed))
    charts = condensation_charts(history, snapshots, len(snapshots)-1)
    return (*charts, gr.update(maximum=len(snapshots)-1, value=len(snapshots)-1), (history, snapshots))


def select_stochastic(index, state):
    if state is None:
        return gr.skip(), gr.skip(), gr.skip(), "Run the simulation first."
    return condensation_charts(*state, int(index))


def run_rl(population, initial, wager, episodes, horizon, trials, seed, allow_skip):
    models = train_all(n=int(population), horizon=int(horizon), episodes=int(episodes),
                       initial=float(initial), other_bet=float(wager)/100,
                       seed=int(seed), allow_skip=bool(allow_skip))
    results = evaluate(models, n=int(population), horizon=int(horizon),
                       initial=float(initial), other_bet=float(wager)/100,
                       seed=int(seed)+123456, trials=int(trials))
    return (*rl_charts(models, results, int(population), int(horizon), float(initial)), models)


def run_animation(models, population, initial, wager, horizon, interval, seed, policy):
    if policy == 'rl' and models is None:
        raise gr.Error("Train Walter in tab 2 first, or choose Fixed, Random, or Aggressive Walter.")
    # The simulation function expects a model dictionary even for non-RL policies.
    # If no trained model is available, create a zero Q-table placeholder without training.
    if models is None:
        models = {'Walter': Trained('Walter', 'Concentrator', np.zeros(Q_SHAPE),
                                    np.zeros(1), True)}
    simulation = simulate_condensation(
        models, n=int(population), initial=float(initial), other_bet=float(wager)/100,
        horizon=int(horizon), snapshot_every=int(interval), seed=int(seed),
        walter_policy=policy)
    figures = condensation_animation_charts(simulation, 0)
    return (simulation, gr.update(minimum=0, maximum=len(simulation['trades'])-1, value=0),
            *figures, False, gr.update(value='▶ Play'), gr.update(active=False))


def show_animation_frame(index, simulation):
    if simulation is None:
        return (gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                'Run the shared-economy simulation first.')
    return condensation_animation_charts(simulation, int(index))


def toggle_play(playing, simulation, index):
    if simulation is None:
        raise gr.Error('Run the simulation first.')
    new_playing = not bool(playing)
    if new_playing and int(index) >= len(simulation['trades'])-1:
        index = 0
    return (new_playing, gr.update(value='Ⅱ Pause' if new_playing else '▶ Play'),
            gr.update(active=new_playing), gr.update(value=index))


def tick_animation(index, simulation, playing):
    if simulation is None or not playing:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    next_index = min(int(index)+1, len(simulation['trades'])-1)
    still_playing = next_index < len(simulation['trades'])-1
    return (gr.update(value=next_index), still_playing,
            gr.update(value='Ⅱ Pause' if still_playing else '▶ Play'),
            gr.update(active=still_playing))


def reset_animation():
    return (gr.update(value=0), False, gr.update(value='▶ Play'), gr.update(active=False))




# Separate three-person showdown: no background agents.
SHOWDOWN_NAMES = ("Walter", "Gustavo", "Jesse")
SHOWDOWN_COLORS = ("#C47B58", "#6B927C", "#9285AD")
SHOWDOWN_PAPER = "#F7F4EA"


def _showdown_style(fig, title, height=360):
    fig.update_layout(
        title=dict(text=title, x=0.02),
        paper_bgcolor=SHOWDOWN_PAPER, plot_bgcolor=SHOWDOWN_PAPER,
        font=dict(color="#292620", family="Arial"),
        height=height, margin=dict(l=55, r=25, t=75, b=50),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(41,38,32,.12)", rangemode="tozero")
    return fig


def showdown_frame(index, simulation):
    if simulation is None:
        return gr.skip(), gr.skip(), "Run the three-character showdown first."
    snapshots = np.asarray(simulation["snapshots"], dtype=float)
    trades = np.asarray(simulation["trades"], dtype=int)
    index = max(0, min(int(index), len(trades) - 1))
    wealth = snapshots[index]
    tick = int(trades[index])

    bars = go.Figure(go.Bar(
        x=SHOWDOWN_NAMES, y=wealth, marker_color=SHOWDOWN_COLORS,
        text=[f"${v:,.2f}" for v in wealth], textposition="outside",
        hovertemplate="%{x}: $%{y:,.2f}<extra></extra>",
    ))
    bars.update_yaxes(title="Wealth ($)", range=[0, max(float(snapshots.max()) * 1.12, 1)])
    _showdown_style(bars, f"The Bread Wars · trade {tick:,}", 400)

    paths = go.Figure()
    for idx, name in enumerate(SHOWDOWN_NAMES):
        paths.add_scatter(
            x=trades[:index + 1], y=snapshots[:index + 1, idx],
            name=name, mode="lines", line=dict(color=SHOWDOWN_COLORS[idx], width=3),
        )
    paths.update_xaxes(title="Trades", range=[0, max(int(trades[-1]), 1)])
    paths.update_yaxes(title="Wealth ($)")
    _showdown_style(paths, "Wealth over the showdown", 350)

    rank = np.argsort(-wealth, kind="stable")
    leaders = ", ".join(SHOWDOWN_NAMES[i] for i in np.flatnonzero(np.isclose(wealth, wealth.max())))
    stats = (
        f"### Trade {tick:,} · Snapshot {index + 1}/{len(trades)}\n\n"
        + "  |  ".join(f"**{name}:** ${wealth[i]:,.2f}" for i, name in enumerate(SHOWDOWN_NAMES))
        + f"\n\n**Current leader:** {leaders}  |  **Total wealth:** ${wealth.sum():,.2f}"
    )
    if index == len(trades) - 1:
        stats += f"\n\n**Final winner:** {simulation['winner']}"
    return bars, paths, stats


def run_showdown_training(episodes, horizon, initial, fixed_pct, trials, seed, allow_skip):
    model = train_showdown(
        episodes=int(episodes), horizon=int(horizon), initial=float(initial),
        fixed_bet=float(fixed_pct) / 100, seed=int(seed), allow_skip=bool(allow_skip),
    )
    evaluation = evaluate_showdown(model, trials=int(trials), seed=int(seed) + 10000)
    curve = np.asarray(model["training_win_rate"])
    training_plot = go.Figure()
    training_plot.add_scatter(x=np.arange(1, len(curve) + 1), y=curve,
                              line=dict(color=SHOWDOWN_COLORS[0], width=3),
                              name="Walter first-place rate")
    training_plot.update_xaxes(title="Training episodes")
    training_plot.update_yaxes(title="Rolling first-place rate", range=[0, 1], tickformat=".0%")
    _showdown_style(training_plot, "Walter · learning to finish first")

    evaluation_plot = go.Figure()
    for i, name in enumerate(SHOWDOWN_NAMES):
        evaluation_plot.add_bar(
            x=["Walter RL", "Walter fixed", "Walter random", "Walter 100%"],
            y=[100 * evaluation[p][name] for p in ("rl", "baseline", "random", "aggressive")],
            name=name, marker_color=SHOWDOWN_COLORS[i],
        )
    evaluation_plot.update_layout(barmode="group")
    evaluation_plot.update_yaxes(title="Strict first-place finishes (%)", range=[0, 100])
    _showdown_style(evaluation_plot, f"Who wins? · {int(trials)} trials per strategy", 380)
    return model, training_plot, evaluation_plot, "**Walter trained.** Choose *Walter Q-learning* below and run the showdown."


def run_showdown_animation(model, horizon, initial, fixed_pct, interval, seed, policy):
    if policy == "rl" and model is None:
        raise gr.Error("Train Walter in this showdown tab first, or choose a non-RL strategy.")
    if model is not None and policy == "rl":
        if (int(horizon) != model["horizon"] or float(initial) != model["initial"]
                or not np.isclose(float(fixed_pct) / 100, model["fixed_bet"])):
            raise gr.Error("Showdown parameters differ from training. Retrain Walter with these settings.")
    simulation = simulate_showdown(
        model=model, horizon=int(horizon), initial=float(initial),
        fixed_bet=float(fixed_pct) / 100, snapshot_every=int(interval),
        seed=int(seed), walter_policy=policy,
    )
    return (
        simulation,
        gr.update(minimum=0, maximum=len(simulation["trades"]) - 1, value=0),
        *showdown_frame(0, simulation),
        False, gr.update(value="▶ Play"), gr.update(active=False),
    )


with gr.Blocks(title='Yard Sale Model · Interactive Lab', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# 🍞 Yard Sale Model\n**Fair trades. Unfair outcomes.** · Interactive Computational Economics')
    trained_models = gr.State(None)

    with gr.Tabs():
        with gr.Tab('1 · Wealth Condensation'):
            gr.Markdown('### Can fair coin-flip trades concentrate almost all the wealth?')
            with gr.Row():
                n = gr.Slider(10, 300, value=100, step=10, label='Agents')
                initial = gr.Number(value=100, minimum=1, label='Starting wealth')
                wager = gr.Slider(1, 50, value=20, step=1, label='Wager (% of poorer wealth)')
            with gr.Row():
                trades = gr.Slider(1000, 1_000_000, value=100_000, step=1000, label='Trades')
                seed = gr.Number(value=42, precision=0, label='Seed')
            run = gr.Button('Run stochastic simulation', variant='primary')
            frame = gr.Slider(0, 100, value=100, step=1, label='Explore snapshots')
            stochastic_state = gr.State(None)
            with gr.Row():
                c1 = gr.Plot()
                c2 = gr.Plot()
            c3 = gr.Plot()
            summary = gr.Markdown()
            run.click(run_stochastic, [n, initial, wager, trades, seed],
                      [c1, c2, c3, summary, frame, stochastic_state])
            frame.change(select_stochastic, [frame, stochastic_state], [c1, c2, c3, summary])

        with gr.Tab('2 · Walter vs Gustavo vs Jesse · RL'):
            gr.Markdown('### Three characters in one economy\n'
                        '**Walter:** Q-learning, trains for richest 1% (25% share is a stricter diagnostic). '
                        '**Gustavo:** fixed wager, top 10% goal. '
                        '**Jesse:** random wagers, ≥150% starting wealth goal. '
                        'Only Walter learns.')
            with gr.Row():
                rn = gr.Slider(20, 200, value=100, step=10, label='Population')
                ri = gr.Number(value=100, minimum=1, label='Initial wealth')
                rb = gr.Slider(2, 50, value=20, step=1, label="Other agents' wager (%)")
            with gr.Row():
                episodes = gr.Slider(100, 20000, value=500, step=100, label='Training episodes')
                horizon = gr.Slider(500, 10000, value=5000, step=500, label='Total trades per episode')
                trials = gr.Slider(50, 2000, value=500, step=50, label='Evaluation episodes')
                rseed = gr.Number(value=42, precision=0, label='Seed')
            allow_skip = gr.Checkbox(value=True, label='Allow 0% wager (skip)')
            train = gr.Button('Train Walter; evaluate all three in one economy', variant='primary')
            gr.Markdown('**Note:** A 0% wager consumes a selected trade without transferring wealth. '
                        'Training may take several minutes or longer.')
            with gr.Row():
                training = gr.Plot()
                comparison = gr.Plot()
            paths = gr.Plot()
            policies = gr.Plot()
            results_text = gr.Markdown()
            # The plotting function returns training, paths, policies, comparison, summary.
            train.click(run_rl, [rn, ri, rb, episodes, horizon, trials, rseed, allow_skip],
                        [training, paths, policies, comparison, results_text, trained_models])

        with gr.Tab('3 · Watch Wealth Condense'):
            gr.Markdown('### Watch inequality emerge from fair trades\n'
                        'Run a **single shared economy**, automatically capture snapshots, '
                        'then play, pause, or scrub through the recorded wealth distribution. '
                        'To use Walter’s learned policy, train him in tab 2 first. '
                        'A policy trained at a different horizon may not generalize.')
            with gr.Row():
                an = gr.Slider(20, 300, value=100, step=10, label='Agents')
                ai = gr.Number(value=100, minimum=1, label='Starting wealth')
                ab = gr.Slider(1, 50, value=20, step=1, label='Background wager (%)')
            with gr.Row():
                ah = gr.Slider(1000, 500_000, value=100_000, step=1000, label='Total trades')
                ae = gr.Slider(100, 10000, value=1000, step=100, label='Trades between snapshots')
                aseed = gr.Number(value=42, precision=0, label='Seed')
                apolicy = gr.Dropdown(choices=[('Walter Q-learning', 'rl'),
                                               ('Walter fixed wager', 'baseline'),
                                               ('Walter random wager', 'random'),
                                               ('Walter always 100%', 'aggressive')],
                                      value='baseline', label='Walter strategy')
            arun = gr.Button('Run economy and capture snapshots', variant='primary')
            astate = gr.State(None)
            playing = gr.State(False)
            with gr.Row():
                play = gr.Button('▶ Play')
                reset = gr.Button('↺ Reset')
                aframe = gr.Slider(0, 100, value=0, step=1, label='Snapshot')
            timer = gr.Timer(value=0.15, active=False)
            with gr.Row():
                inequality = gr.Plot()
                distribution = gr.Plot()
            characters = gr.Plot()
            lorenz = gr.Plot()
            astats = gr.Markdown('Run the simulation to explore its wealth distribution.')

            arun.click(run_animation, [trained_models, an, ai, ab, ah, ae, aseed, apolicy],
                       [astate, aframe, inequality, distribution, characters, lorenz,
                        astats, playing, play, timer])
            aframe.change(show_animation_frame, [aframe, astate],
                          [inequality, distribution, characters, lorenz, astats])
            play.click(toggle_play, [playing, astate, aframe], [playing, play, timer, aframe])
            reset.click(reset_animation, outputs=[aframe, playing, play, timer])
            timer.tick(tick_animation, [aframe, astate, playing], [aframe, playing, play, timer])


        with gr.Tab('4 · Walter vs Gustavo vs Jesse · Showdown'):
            gr.Markdown(
                '### The Bread Wars · Three-character showdown\n'
                '**Only Walter, Gustavo, and Jesse trade.** Each begins with the same wealth. '
                'Walter learns to finish strictly first; Gustavo uses a fixed wager; Jesse randomizes. '
                'Both selected agents propose wagers, and the average determines the fair-trade stake.'
            )
            showdown_model = gr.State(None)
            showdown_state = gr.State(None)
            showdown_playing = gr.State(False)
            with gr.Row():
                sh_initial = gr.Number(value=100, minimum=1, label='Starting wealth per character ($)')
                sh_bet = gr.Slider(1, 50, value=20, step=1, label='Gustavo fixed wager (%)')
                sh_seed = gr.Number(value=42, precision=0, label='Seed')
            with gr.Row():
                sh_episodes = gr.Slider(100, 10000, value=2000, step=100, label='Walter training episodes')
                sh_horizon = gr.Slider(50, 2000, value=300, step=50, label='Trades per showdown')
                sh_trials = gr.Slider(50, 2000, value=300, step=50, label='Evaluation showdowns per strategy')
            sh_skip = gr.Checkbox(value=True, label='Allow Walter to propose a 0% wager')
            sh_train = gr.Button('Train Walter and compare win rates', variant='primary')
            with gr.Row():
                sh_training_plot = gr.Plot(label='Training success')
                sh_evaluation_plot = gr.Plot(label='Showdown win rates')
            sh_train_stats = gr.Markdown('Train Walter to compare strategies, or run a fixed-wager showdown below.')
            with gr.Row():
                sh_interval = gr.Slider(1, 100, value=5, step=1, label='Trades between snapshots')
                sh_policy = gr.Dropdown(
                    choices=[('Walter Q-learning', 'rl'),
                             ('Walter fixed wager', 'baseline'),
                             ('Walter random wager', 'random'),
                             ('Walter always 100%', 'aggressive')],
                    value='baseline', label='Walter showdown strategy',
                )
            sh_run = gr.Button('Run three-character showdown', variant='primary')
            with gr.Row():
                sh_play = gr.Button('▶ Play')
                sh_reset = gr.Button('↺ Reset')
                sh_frame = gr.Slider(0, 60, value=0, step=1, label='Showdown snapshot')
            sh_timer = gr.Timer(value=0.15, active=False)
            sh_bars = gr.Plot(label='Walter vs Gustavo vs Jesse · live wealth')
            sh_paths = gr.Plot(label='Wealth trajectories')
            sh_stats = gr.Markdown('Run a showdown to see the three wealth bars.')

            sh_train.click(
                run_showdown_training,
                [sh_episodes, sh_horizon, sh_initial, sh_bet, sh_trials, sh_seed, sh_skip],
                [showdown_model, sh_training_plot, sh_evaluation_plot, sh_train_stats],
            )
            sh_run.click(
                run_showdown_animation,
                [showdown_model, sh_horizon, sh_initial, sh_bet, sh_interval, sh_seed, sh_policy],
                [showdown_state, sh_frame, sh_bars, sh_paths, sh_stats,
                 showdown_playing, sh_play, sh_timer],
            )
            sh_frame.change(showdown_frame, [sh_frame, showdown_state],
                            [sh_bars, sh_paths, sh_stats])
            sh_play.click(toggle_play, [showdown_playing, showdown_state, sh_frame],
                          [showdown_playing, sh_play, sh_timer, sh_frame])
            sh_reset.click(reset_animation, outputs=[sh_frame, showdown_playing, sh_play, sh_timer])
            sh_timer.tick(tick_animation, [sh_frame, showdown_state, showdown_playing],
                          [sh_frame, showdown_playing, sh_play, sh_timer])

    gr.Markdown('**Scientific scope:** Closed economy, fair independent outcomes, '
                'no productivity, borrowing, taxation, or income. '
                'This is a stylized model, not a real-world economic forecast.')

if __name__ == '__main__':
    demo.queue().launch(server_port=7860)