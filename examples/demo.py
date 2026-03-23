"""Demo script showing how to use the Poker CFR Solver.

This script trains all three algorithms on Kuhn Poker and displays
the resulting Nash equilibrium strategies.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports work when running from examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.cfr import VanillaCFR
from src.core.cfr_plus import CFRPlus
from src.core.mccfr import MCCFR
from src.core.exploitability import compute_exploitability
from src.games.kuhn_poker import KuhnPoker


def print_strategy(strategy: dict[str, np.ndarray], action_names: list[str]) -> None:
    """Pretty-print a strategy profile."""
    for key in sorted(strategy.keys()):
        probs = strategy[key]
        parts = [f"{action_names[i]}={p:.4f}" for i, p in enumerate(probs)]
        print(f"  {key:20s} -> {', '.join(parts)}")


def main() -> None:
    print("=" * 60)
    print("  Poker CFR Solver — Demo")
    print("=" * 60)

    game = KuhnPoker()
    num_iterations = 10000
    action_names = game.action_names()

    # ---- Vanilla CFR ----
    print(f"\n--- Vanilla CFR ({num_iterations} iterations) ---")
    solver = VanillaCFR(game)
    history = solver.train(num_iterations)
    strategy = solver.average_strategy()

    print(f"\nNash Equilibrium Strategy:")
    print_strategy(strategy, action_names)

    expl = compute_exploitability(game, strategy)
    print(f"\nExploitability: {expl:.8f}")
    print(f"Information sets: {len(solver.strategy_map)}")

    # ---- CFR+ ----
    print(f"\n--- CFR+ ({num_iterations} iterations) ---")
    solver_plus = CFRPlus(game)
    history_plus = solver_plus.train(num_iterations)
    strategy_plus = solver_plus.average_strategy()

    expl_plus = compute_exploitability(game, strategy_plus)
    print(f"Exploitability: {expl_plus:.8f}")

    # ---- MCCFR ----
    print(f"\n--- MCCFR ({num_iterations} iterations) ---")
    solver_mc = MCCFR(game, seed=42)
    history_mc = solver_mc.train(num_iterations)
    strategy_mc = solver_mc.average_strategy()

    expl_mc = compute_exploitability(game, strategy_mc)
    print(f"Exploitability: {expl_mc:.8f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"{'Algorithm':<15} {'Exploitability':<20}")
    print("-" * 35)
    print(f"{'Vanilla CFR':<15} {expl:<20.8f}")
    print(f"{'CFR+':<15} {expl_plus:<20.8f}")
    print(f"{'MCCFR':<15} {expl_mc:<20.8f}")

    print("\nKuhn Poker Nash Equilibrium Properties:")
    print("  - Player 0 with Jack: bluffs with probability alpha in [0, 1/3]")
    print("  - Player 0 with King: always bets")
    print("  - Player 1 with King facing bet: always calls")
    print("  - Player 1 with Jack facing bet: always folds")
    print(f"\nVerification (from Vanilla CFR):")
    print(f"  J: pass bet={strategy['J:'][0]:.4f} {strategy['J:'][1]:.4f}")
    print(f"  K: pass bet={strategy['K:'][0]:.4f} {strategy['K:'][1]:.4f}")
    print(f"  K:b pass call={strategy['K:b'][0]:.4f} {strategy['K:b'][1]:.4f}")
    print(f"  J:b fold call={strategy['J:b'][0]:.4f} {strategy['J:b'][1]:.4f}")


if __name__ == "__main__":
    main()
