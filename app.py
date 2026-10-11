"""Launch: python app.py  |  http://127.0.0.1:7860"""
import gradio as gr
from src.economy import simulate
from src.rl import train_all, evaluate
from src.plots import condensation_charts, rl_charts

with gr.Blocks(title="Yard Sale Model · Interactive Lab") as demo:
    gr.Markdown("# 🍞 Yard Sale Model\n**Fair trades. Unfair outcomes.** · Interactive Computational Economics")
    with gr.Tabs():
        with gr.Tab("1 · Wealth Condensation"):
            gr.Markdown("### Can fair coin-flip trades concentrate almost all the wealth?")
            with gr.Row():
                n=gr.Slider(10,300,value=100,step=10,label="Agents")
                initial=gr.Number(value=100,minimum=1,label="Starting wealth")
                wager=gr.Slider(1,50,value=20,step=1,label="Wager (% of poorer wealth)")
            with gr.Row():
                trades=gr.Slider(1000,1_000_000,value=100_000,step=1000,label="Trades")
                seed=gr.Number(value=42,precision=0,label="Seed")
            run=gr.Button("Run stochastic simulation",variant="primary")
            frame=gr.Slider(0,100,value=100,step=1,label="Explore snapshots")
            state=gr.State(None)
            with gr.Row():
                c1=gr.Plot()
                c2=gr.Plot()
            c3=gr.Plot()
            summary=gr.Markdown()
            def run_stochastic(n,initial,wager,trades,seed):
                hist,snaps=simulate(int(n),float(initial),float(wager)/100,int(trades),int(seed))
                charts=condensation_charts(hist,snaps,len(snaps)-1)
                return (*charts,gr.update(maximum=len(snaps)-1,value=len(snaps)-1),(hist,snaps))
            def select_frame(frame,state):
                if state is None:
                    return gr.skip(),gr.skip(),gr.skip(),"Run the simulation first."
                return condensation_charts(*state,int(frame))
            run.click(run_stochastic,[n,initial,wager,trades,seed],
                      [c1,c2,c3,summary,frame,state])
            frame.change(select_frame,[frame,state],[c1,c2,c3,summary])
        with gr.Tab("2 · Can a Genius Beat It? · RL"):
            gr.Markdown("### Three agents, three goals\n"
                "**Gambler:** top 1% · **Comfortable:** above median · "
                "**Survivor:** outside bottom 10%. "
                "One learner chooses its stake; all other agents bet a fixed percentage.")
            with gr.Row():
                rn=gr.Slider(20,200,value=100,step=10,label="Population")
                ri=gr.Number(value=100,minimum=1,label="Initial wealth")
                rb=gr.Slider(2,50,value=20,step=1,label="Other agents' wager (%)")
            with gr.Row():
                episodes=gr.Slider(50,3000,value=400,step=50,label="Training episodes")
                horizon=gr.Slider(50,1500,value=300,step=50,label="Rounds per episode")
                trials=gr.Slider(30,500,value=100,step=10,label="Evaluation episodes")
                rseed=gr.Number(value=42,precision=0,label="Seed")
            allow_skip=gr.Checkbox(value=True,label="Allow 0% wager (skip)")
            train=gr.Button("Train and evaluate all three agents",variant="primary")
            gr.Markdown("**Note:** training is tabular Monte Carlo reinforcement learning; "
                        "it may take a while with high episode/round counts. "
                        "Evaluation uses new random streams.")
            with gr.Row():
                training=gr.Plot()
                comparison=gr.Plot()
            paths=gr.Plot()
            policies=gr.Plot()
            results_text=gr.Markdown()
            def run_rl(rn,ri,rb,episodes,horizon,trials,rseed,allow_skip):
                models=train_all(int(rn),int(horizon),int(episodes),float(ri),
                                 float(rb)/100,int(rseed),bool(allow_skip))
                results=evaluate(models,int(rn),int(horizon),float(ri),
                                 float(rb)/100,int(rseed)+123456,int(trials))
                return rl_charts(models,results,int(rn),int(horizon),float(ri))
            train.click(run_rl,[rn,ri,rb,episodes,horizon,trials,rseed,allow_skip],
                        [training,paths,policies,comparison,results_text])
    gr.Markdown("**Scientific scope:** closed economy, fair independent outcomes, "
                "no productivity, borrowing, taxation, or income. "
                "This is a stylized model, not a real-world economic forecast.")

if __name__=="__main__":
    demo.queue().launch(server_port=7860)
