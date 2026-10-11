"""Reusable parchment-themed Plotly figures."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PAPER="#F7F4EA"
INK="#292620"
GREEN="#6B927C"
ORANGE="#C47B58"
BLUE="#6588AA"
PALETTE={"Gambler":ORANGE,"Comfortable":BLUE,"Survivor":GREEN}

def style(fig, title, height=350):
    fig.update_layout(title=title, paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(color=INK, family="Arial"), height=height,
        margin=dict(l=48,r=25,t=60,b=45),
        legend=dict(orientation="h",y=1.12,x=0))
    fig.update_xaxes(showgrid=False, linecolor=INK)
    fig.update_yaxes(gridcolor="rgba(41,38,32,.12)",zeroline=False)
    return fig

def condensation_charts(history, snapshots, index):
    index = int(np.clip(index, 0, len(snapshots)-1))
    wealth = snapshots[index]
    tick=int(history[index,0])
    timeline=go.Figure()
    timeline.add_scatter(x=history[:,0],y=history[:,1],name="Gini",line=dict(color=ORANGE,width=3))
    timeline.add_scatter(x=history[:,0],y=history[:,2],name="Richest share",line=dict(color=GREEN,width=3))
    timeline.add_vline(x=tick,line_dash="dot",line_color=INK)
    timeline.update_yaxes(range=[0,1])
    timeline.update_xaxes(title="Trades")
    style(timeline,"Inequality over time")
    distribution=go.Figure(go.Bar(x=np.arange(1,len(wealth)+1),y=np.sort(wealth)[::-1],
                                 marker_color=[ORANGE]+[GREEN]*(len(wealth)-1)))
    distribution.update_xaxes(title="Wealth rank (richest first)")
    distribution.update_yaxes(title="Wealth")
    style(distribution,f"Wealth distribution · trade {tick:,}")
    sorted_w=np.sort(wealth)
    lorenz=go.Figure()
    lorenz.add_scatter(x=[0,1],y=[0,1],name="Perfect equality",
                       line=dict(color=INK,dash="dash"))
    lorenz.add_scatter(x=np.linspace(0,1,len(wealth)+1),
                       y=np.r_[0,np.cumsum(sorted_w)/sorted_w.sum()],
                       name="Observed",line=dict(color=ORANGE,width=3))
    lorenz.update_xaxes(title="Cumulative population share",range=[0,1])
    lorenz.update_yaxes(title="Cumulative wealth share",range=[0,1])
    style(lorenz,"Lorenz curve")
    top_n=max(1,int(np.ceil(.1*len(wealth))))
    stats=(f"### Snapshot · {tick:,} trades\n"
           f"**Gini:** {history[index,1]:.3f}  |  "
           f"**Richest agent:** {100*wealth.max()/wealth.sum():.1f}%  |  "
           f"**Top 10%:** {100*np.sort(wealth)[-top_n:].sum()/wealth.sum():.1f}%  |  "
           f"**Total wealth:** {wealth.sum():,.0f}")
    return timeline,distribution,lorenz,stats

def rl_charts(models, results, n, horizon, initial):
    training=make_subplots(rows=1,cols=3,subplot_titles=list(models))
    paths=make_subplots(rows=1,cols=3,subplot_titles=list(models))
    policies=make_subplots(rows=1,cols=3,subplot_titles=list(models))
    for col,(name,model) in enumerate(models.items(),1):
        color=PALETTE[name]
        training.add_scatter(x=np.arange(1,len(model.curve)+1),y=model.curve,
                             line=dict(color=color,width=2),name=name,
                             showlegend=False,row=1,col=col)
        paths_data=results[name]["paths"]
        for path in paths_data:
            paths.add_scatter(x=np.arange(len(path)),y=path,mode="lines",
                              line=dict(color="rgba(90,90,90,.18)",width=1),
                              showlegend=False,row=1,col=col)
        if len(paths_data):
            paths.add_scatter(x=np.arange(paths_data.shape[1]),y=np.median(paths_data,axis=0),
                              line=dict(color=color,width=3),name=name,
                              showlegend=False,row=1,col=col)
        # At final time phase, policy vs relative-wealth-to-median state.
        q=model.q[:,-1,:]
        action_indices=np.argmax(q,axis=-1)
        from .rl import ACTIONS
        policies.add_scatter(x=np.arange(len(action_indices)),y=ACTIONS[action_indices]*100,
                             mode="lines+markers",line=dict(color=color,width=2),
                             showlegend=False,row=1,col=col)
    style(training,"Training success (rolling average)",height=340)
    training.update_yaxes(range=[0,1],title="Success fraction")
    training.update_xaxes(title="Training episode")
    style(paths,"Evaluation wealth paths (gray) and median (color)",height=340)
    paths.update_xaxes(title="Rounds")
    paths.update_yaxes(title="Agent wealth")
    style(policies,"Learned wager by wealth-state bin (late episode)",height=340)
    policies.update_xaxes(title="Relative-wealth state bin")
    policies.update_yaxes(title="Wager (% of poorer wealth)",range=[0,100])
    compare=go.Figure()
    names=list(models)
    compare.add_bar(x=names,y=[100*results[k]["rl"] for k in names],name="RL",
                    marker_color=[PALETTE[k] for k in names])
    compare.add_bar(x=names,y=[100*results[k]["random"] for k in names],name="Baseline 20% bettor",
                    marker_color="#999999")
    compare.update_layout(barmode="group")
    compare.update_yaxes(title="Success rate (%)",range=[0,100])
    style(compare,"Out-of-sample success rates",height=370)
    summary="### Evaluation results\n\n| Goal | RL | Baseline | Trials |\n|---|---:|---:|---:|\n"
    for name in names:
        r=results[name]
        summary+=f"| {name} | {r['rl']*100:.1f}% | {r['random']*100:.1f}% | {r['trials']} |\n"
    summary+=("\nThe baseline always wagers **20%** (or the configured other-agent wager). "
              "Success rates are estimates from independent evaluation episodes, not guaranteed advantages. "
              "Training curves show *rolling terminal success*, not a smooth theoretical learning curve. "
              "Top 1% is rounded up to at least one position.")
    return training,paths,policies,compare,summary
