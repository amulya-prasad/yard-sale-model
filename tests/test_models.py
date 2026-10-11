import numpy as np
from yard_sale_model.economy import simulate, gini
from yard_sale_model.rl import train_all, evaluate, succeeded

def test_conservation():
    h, s=simulate(n=20,initial=100,fraction=.2,trades=3000,seed=7)
    assert np.allclose(s.sum(axis=1),2000)
    assert np.isclose(gini(s[0]),0)
    assert h[-1,1] >= 0

def test_reproducibility():
    _,a=simulate(n=10,trades=1000,seed=8)
    _,b=simulate(n=10,trades=1000,seed=8)
    assert np.array_equal(a,b)

def test_goals():
    w=np.array([100,0,0,0,0,0,0,0,0,0],dtype=float)
    assert succeeded(w,"Gambler")==1
    assert succeeded(w,"Comfortable")==1
    assert succeeded(w,"Survivor")==1

def test_rl_smoke():
    m=train_all(n=20,horizon=10,episodes=3,seed=5)
    e=evaluate(m,n=20,horizon=10,trials=3,seed=9)
    assert set(e)==set(m)
    assert all(0<=e[k]["rl"]<=1 for k in e)
