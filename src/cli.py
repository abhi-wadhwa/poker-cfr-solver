"""Command-line interface for the Poker CFR Solver.

Usage:
    python -m src.cli train --game kuhn --algo cfr --iterations 10000
    python -m src.cli train --game leduc --algo cfr-plus --iterations 5000
    python -m src.cli show --game kuhn --algo cfr --iterations 10000
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog
import typer

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.cfr import VanillaCFR
from src.core.cfr_plus import CFRPlus
from src.core.exploitability import compute_exploitability
from src.core.mccfr import MCCFR
from src.games.kuhn_poker import KuhnPoker
from src.games.leduc_holdem import LeducHoldem

logger = structlog.get_logger()

app = typer.Typer(
    name="poker-cfr",
    help="Poker CFR Solver — compute Nash equilibria for poker games.",
    add_completion=False,
)


def _create_game(game_name: str):
    """Create a game instance from its name."""
    if game_name == "kuhn":
        return KuhnPoker()
    elif game_name == "leduc":
        return LeducHoldem()
    else:
        typer.echo(f"Unknown game: {game_name}. Choose 'kuhn' or 'leduc'.")
        raise typer.Exit(1)


def _create_solver(algo_name: str, game, seed: int = 42):
    """Create a solver instance from its name."""
    if algo_name == "cfr":
        return VanillaCFR(game)
    elif algo_name == "cfr-plus":
        return CFRPlus(game)
    elif algo_name == "mccfr":
        return MCCFR(game, seed=seed)
    else:
        typer.echo(f"Unknown algorithm: {algo_name}. Choose 'cfr', 'cfr-plus', or 'mccfr'.")
        raise typer.Exit(1)


@app.command()
def train(
    game: str = typer.Option("kuhn", help="Game to solve: 'kuhn' or 'leduc'"),
    algo: str = typer.Option("cfr", help="Algorithm: 'cfr', 'cfr-plus', or 'mccfr'"),
    iterations: int = typer.Option(10000, help="Number of training iterations"),
    seed: int = typer.Option(42, help="Random seed for MCCFR"),
    verbose: bool = typer.Option(False, help="Print detailed progress"),
) -> None:
    """Train a CFR solver on the specified game."""
    typer.echo(f"Training {algo.upper()} on {game.capitalize()} for {iterations} iterations...")

    game_instance = _create_game(game)
    solver = _create_solver(algo, game_instance, seed)

    exploitability_history = solver.train(iterations)

    final_expl = exploitability_history[-1] if exploitability_history else float("inf")
    typer.echo("\nTraining complete!")
    typer.echo(f"  Final exploitability: {final_expl:.8f}")
    typer.echo(f"  Information sets:     {len(solver.strategy_map)}")

    if verbose:
        typer.echo("\nExploitability over training:")
        for i, expl in enumerate(exploitability_history):
            typer.echo(f"  Checkpoint {i + 1}: {expl:.8f}")


@app.command()
def show(
    game: str = typer.Option("kuhn", help="Game to solve: 'kuhn' or 'leduc'"),
    algo: str = typer.Option("cfr", help="Algorithm: 'cfr', 'cfr-plus', or 'mccfr'"),
    iterations: int = typer.Option(10000, help="Number of training iterations"),
    seed: int = typer.Option(42, help="Random seed for MCCFR"),
) -> None:
    """Train and display the Nash equilibrium strategy."""
    typer.echo(f"Computing Nash equilibrium for {game.capitalize()} using {algo.upper()}...")

    game_instance = _create_game(game)
    solver = _create_solver(algo, game_instance, seed)
    solver.train(iterations)

    strategy = solver.average_strategy()
    action_names = game_instance.action_names()

    typer.echo(f"\nNash Equilibrium Strategy ({len(strategy)} information sets):")
    typer.echo("-" * 60)

    for key in sorted(strategy.keys()):
        probs = strategy[key]
        parts = []
        for i, p in enumerate(probs):
            name = action_names[i] if i < len(action_names) else f"a{i}"
            parts.append(f"{name}={p:.4f}")
        typer.echo(f"  {key:20s} -> {', '.join(parts)}")

    final_expl = compute_exploitability(game_instance, strategy)
    typer.echo(f"\nExploitability: {final_expl:.8f}")


@app.command()
def benchmark(
    game: str = typer.Option("kuhn", help="Game to benchmark"),
    iterations: int = typer.Option(10000, help="Iterations for each algorithm"),
) -> None:
    """Benchmark all three algorithms on a game."""
    import time

    game_instance = _create_game(game)

    results = []
    for algo_name, solver_cls in [
        ("Vanilla CFR", VanillaCFR),
        ("CFR+", CFRPlus),
        ("MCCFR", MCCFR),
    ]:
        if algo_name == "MCCFR":
            solver = solver_cls(game_instance, seed=42)
        else:
            solver = solver_cls(game_instance)

        start_time = time.perf_counter()
        history = solver.train(iterations)
        elapsed = time.perf_counter() - start_time

        final_expl = history[-1] if history else float("inf")
        results.append((algo_name, elapsed, final_expl, len(solver.strategy_map)))

    typer.echo(f"\nBenchmark Results — {game.capitalize()} ({iterations} iterations)")
    typer.echo("=" * 70)
    typer.echo(f"{'Algorithm':<15} {'Time (s)':<12} {'Exploitability':<18} {'Info Sets':<10}")
    typer.echo("-" * 70)
    for name, elapsed, expl, n_sets in results:
        typer.echo(f"{name:<15} {elapsed:<12.3f} {expl:<18.8f} {n_sets:<10}")


if __name__ == "__main__":
    app()
