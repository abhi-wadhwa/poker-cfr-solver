# poker-cfr-solver

counterfactual regret minimization for poker. the algorithm that solved heads-up limit hold'em and powers every modern poker AI.

## what this is

- **vanilla CFR** — full game tree traversal, accumulate regrets, update strategies
- **CFR+** — regret matching+ with floor at zero. converges faster
- **MCCFR** — monte carlo sampling. scales to larger games
- converges to **nash equilibrium** in two-player zero-sum games

## running it

```bash
pip install -r requirements.txt
python main.py
```

## how it works

maintain a "regret" for each action at each decision point. regret = how much better you would have done playing that action vs your actual mix. play proportional to positive regrets (regret matching). zinkevich 2007 proved: if both players minimize average regret, their average strategies converge to NE. you just play against yourself a million times tracking regrets and the strategy converges. self-play at its purest.
