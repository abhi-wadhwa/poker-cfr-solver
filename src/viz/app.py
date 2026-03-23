"""Streamlit interactive dashboard for the Poker CFR Solver.

Features:
    1. Training Dashboard — train CFR/CFR+/MCCFR on Kuhn or Leduc, view
       exploitability convergence curves and strategy evolution.
    2. Strategy Browser — navigate the game tree and inspect Nash equilibrium
       strategies at each information set.
    3. Play Against Bot — play Kuhn Poker against the trained CFR bot.

Run with: streamlit run src/viz/app.py
"""

from __future__ import annotations

import random

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.core.cfr import VanillaCFR
from src.core.cfr_plus import CFRPlus
from src.core.exploitability import compute_exploitability
from src.core.mccfr import MCCFR
from src.games.kuhn_poker import (
    BET,
    CARD_NAMES,
    JACK,
    KING,
    PASS,
    QUEEN,
    KuhnPoker,
    KuhnState,
)
from src.games.leduc_holdem import LeducHoldem

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Poker CFR Solver",
    page_icon="cards",
    layout="wide",
)

st.title("Poker CFR Solver")
st.markdown(
    "Interactive dashboard for training and exploring Nash equilibrium strategies "
    "computed via Counterfactual Regret Minimization."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Training Dashboard", "Strategy Browser", "Play Against Bot"],
)

# ===== SESSION STATE DEFAULTS ==============================================
if "trained_strategy" not in st.session_state:
    st.session_state.trained_strategy = None
if "exploitability_history" not in st.session_state:
    st.session_state.exploitability_history = []
if "strategy_snapshots" not in st.session_state:
    st.session_state.strategy_snapshots = []
if "training_game" not in st.session_state:
    st.session_state.training_game = None
if "training_algo" not in st.session_state:
    st.session_state.training_algo = None

# Play-against-bot state
if "game_active" not in st.session_state:
    st.session_state.game_active = False
if "bot_strategy" not in st.session_state:
    st.session_state.bot_strategy = None
if "player_card" not in st.session_state:
    st.session_state.player_card = None
if "bot_card" not in st.session_state:
    st.session_state.bot_card = None
if "game_history" not in st.session_state:
    st.session_state.game_history = []
if "game_log" not in st.session_state:
    st.session_state.game_log = []
if "player_score" not in st.session_state:
    st.session_state.player_score = 0
if "bot_score" not in st.session_state:
    st.session_state.bot_score = 0
if "round_result" not in st.session_state:
    st.session_state.round_result = None
if "player_is_p0" not in st.session_state:
    st.session_state.player_is_p0 = True
if "waiting_for_new_hand" not in st.session_state:
    st.session_state.waiting_for_new_hand = False

# ===========================================================================
# PAGE 1: TRAINING DASHBOARD
# ===========================================================================
if page == "Training Dashboard":
    st.header("Training Dashboard")
    st.markdown("Train a CFR algorithm and observe convergence to Nash equilibrium.")

    col1, col2, col3 = st.columns(3)
    with col1:
        game_name = st.selectbox("Game", ["Kuhn Poker", "Leduc Hold'em"])
    with col2:
        algo_name = st.selectbox("Algorithm", ["Vanilla CFR", "CFR+", "MCCFR"])
    with col3:
        num_iters = st.slider("Iterations", 100, 50000, 10000, step=100)

    if st.button("Train", type="primary"):
        game = KuhnPoker() if game_name == "Kuhn Poker" else LeducHoldem()

        if algo_name == "Vanilla CFR":
            solver = VanillaCFR(game)
        elif algo_name == "CFR+":
            solver = CFRPlus(game)
        else:
            solver = MCCFR(game, seed=42)

        progress_bar = st.progress(0)
        status_text = st.empty()

        exploitability_history = []
        strategy_snapshots = []

        # Train in chunks to update progress bar
        chunk_size = max(1, num_iters // 100)
        for chunk_start in range(0, num_iters, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_iters)
            actual_chunk = chunk_end - chunk_start

            # Train chunk
            if algo_name == "Vanilla CFR":
                for _ in range(actual_chunk):
                    solver.iteration += 1
                    for traverser in range(game.num_players):
                        initial_state = game.initial_state()
                        reach_probs = np.ones(game.num_players, dtype=np.float64)
                        solver._cfr(initial_state, traverser, reach_probs)
                        # Update cumulative strategy is done inside _cfr
            elif algo_name == "CFR+":
                for _ in range(actual_chunk):
                    solver.iteration += 1
                    for traverser in range(game.num_players):
                        initial_state = game.initial_state()
                        reach_probs = np.ones(game.num_players, dtype=np.float64)
                        solver._cfr_plus(initial_state, traverser, reach_probs)
                    for node in solver.strategy_map.values():
                        node.clamp_regrets()
            else:
                for _ in range(actual_chunk):
                    solver.iteration += 1
                    for traverser in range(game.num_players):
                        initial_state = game.initial_state()
                        solver._external_sampling(initial_state, traverser)

            avg_strat = solver.average_strategy()
            expl = compute_exploitability(game, avg_strat)
            exploitability_history.append(expl)
            strategy_snapshots.append(dict(avg_strat))

            progress = chunk_end / num_iters
            progress_bar.progress(progress)
            status_text.text(
                f"Iteration {chunk_end}/{num_iters} | "
                f"Exploitability: {expl:.6f}"
            )

        st.session_state.trained_strategy = avg_strat
        st.session_state.exploitability_history = exploitability_history
        st.session_state.strategy_snapshots = strategy_snapshots
        st.session_state.training_game = game_name
        st.session_state.training_algo = algo_name

        st.success(
            f"Training complete! Final exploitability: "
            f"{exploitability_history[-1]:.6f}"
        )

    # Display results if training has been done
    if st.session_state.exploitability_history:
        st.subheader("Exploitability Convergence")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=st.session_state.exploitability_history,
                mode="lines",
                name="Exploitability",
                line=dict(color="#2196F3", width=2),
            )
        )
        fig.update_layout(
            xaxis_title="Training Progress (checkpoints)",
            yaxis_title="Exploitability",
            yaxis_type="log",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show strategy evolution for key information sets
        if st.session_state.training_game == "Kuhn Poker" and st.session_state.strategy_snapshots:
            st.subheader("Strategy Evolution (Key Information Sets)")

            key_info_sets = ["J:", "Q:", "K:", "J:b", "Q:b", "K:b"]
            available_keys = set()
            for snap in st.session_state.strategy_snapshots:
                available_keys.update(snap.keys())
            key_info_sets = [k for k in key_info_sets if k in available_keys]

            for info_set in key_info_sets:
                strategies_over_time = []
                for snap in st.session_state.strategy_snapshots:
                    if info_set in snap:
                        strategies_over_time.append(snap[info_set])

                if strategies_over_time:
                    fig = go.Figure()
                    strats = np.array(strategies_over_time)
                    for action_idx, action_name in enumerate(["Pass", "Bet"]):
                        fig.add_trace(
                            go.Scatter(
                                y=strats[:, action_idx],
                                mode="lines",
                                name=action_name,
                            )
                        )
                    fig.update_layout(
                        title=f"Info Set: {info_set}",
                        xaxis_title="Training Progress",
                        yaxis_title="Probability",
                        yaxis_range=[0, 1],
                        template="plotly_white",
                        height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# PAGE 2: STRATEGY BROWSER
# ===========================================================================
elif page == "Strategy Browser":
    st.header("Strategy Browser")

    if st.session_state.trained_strategy is None:
        st.info("Please train a model first using the Training Dashboard.")
    else:
        strategy = st.session_state.trained_strategy
        game_name = st.session_state.training_game

        st.markdown(f"**Game:** {game_name} | **Algorithm:** {st.session_state.training_algo}")
        st.markdown(f"**Information Sets:** {len(strategy)}")

        # Sort info sets for display
        sorted_keys = sorted(strategy.keys())

        # Filter
        search = st.text_input("Filter information sets", "")
        if search:
            sorted_keys = [k for k in sorted_keys if search.lower() in k.lower()]

        st.markdown(f"Showing {len(sorted_keys)} information sets:")

        if game_name == "Kuhn Poker":
            action_names = ["Pass", "Bet"]
        else:
            action_names = ["Fold", "Check/Call", "Raise"]

        # Display as a table
        for key in sorted_keys:
            strat = strategy[key]
            cols = st.columns([2] + [1] * len(strat))
            cols[0].markdown(f"**`{key}`**")
            for i, prob in enumerate(strat):
                name = action_names[i] if i < len(action_names) else f"Action {i}"
                cols[i + 1].metric(name, f"{prob:.4f}")

# ===========================================================================
# PAGE 3: PLAY AGAINST BOT
# ===========================================================================
elif page == "Play Against Bot":
    st.header("Play Against Bot (Kuhn Poker)")

    # Train bot if not already done
    if st.session_state.bot_strategy is None:
        st.info("Training the bot with 10,000 iterations of CFR...")
        game = KuhnPoker()
        solver = VanillaCFR(game)
        solver.train(10000)
        st.session_state.bot_strategy = solver.average_strategy()
        st.success("Bot trained and ready!")

    bot_strategy = st.session_state.bot_strategy

    def bot_action(card: int, history: tuple[int, ...]) -> int:
        """Choose the bot's action using the trained strategy."""
        card_name = CARD_NAMES[card]
        hist_str = "".join("p" if a == PASS else "b" for a in history)
        info_key = f"{card_name}:{hist_str}"
        if info_key in bot_strategy:
            strat = bot_strategy[info_key]
            r = random.random()
            if r < strat[0]:
                return PASS
            return BET
        # Fallback: random
        return random.choice([PASS, BET])

    def start_new_hand():
        """Deal a new hand."""
        cards = random.sample([JACK, QUEEN, KING], 2)
        st.session_state.player_is_p0 = random.random() < 0.5
        if st.session_state.player_is_p0:
            st.session_state.player_card = cards[0]
            st.session_state.bot_card = cards[1]
        else:
            st.session_state.player_card = cards[1]
            st.session_state.bot_card = cards[0]
        st.session_state.game_history = []
        st.session_state.game_active = True
        st.session_state.round_result = None
        st.session_state.waiting_for_new_hand = False

        # If bot is player 0, bot acts first
        if not st.session_state.player_is_p0:
            action = bot_action(st.session_state.bot_card, ())
            st.session_state.game_history.append(action)
            action_name = "passes" if action == PASS else "bets"
            st.session_state.game_log.append(f"Bot {action_name}")

    def evaluate_hand():
        """Evaluate the current hand and determine winner."""
        h = tuple(st.session_state.game_history)
        p_card = st.session_state.player_card
        b_card = st.session_state.bot_card

        if st.session_state.player_is_p0:
            cards = (p_card, b_card)
            player_idx = 0
        else:
            cards = (b_card, p_card)
            player_idx = 1

        game = KuhnPoker()
        state = KuhnState(cards=cards, history=h)
        if game.is_terminal(state):
            utility = game.terminal_utility(state, player_idx)
            return utility
        return None

    def player_act(action: int):
        """Process the player's action."""
        action_name = "passes" if action == PASS else "bets"
        st.session_state.game_log.append(f"You {action_name}")
        st.session_state.game_history.append(action)

        # Check if game is over
        result = evaluate_hand()
        if result is not None:
            finalize_hand(result)
            return

        # Bot's turn
        h = tuple(st.session_state.game_history)
        bot_act = bot_action(st.session_state.bot_card, h)
        bot_action_name = "passes" if bot_act == PASS else "bets"
        st.session_state.game_log.append(f"Bot {bot_action_name}")
        st.session_state.game_history.append(bot_act)

        # Check if game is over after bot's action
        result = evaluate_hand()
        if result is not None:
            finalize_hand(result)

    def finalize_hand(result: float):
        """End the hand and update scores."""
        st.session_state.game_active = False
        st.session_state.waiting_for_new_hand = True
        if result > 0:
            st.session_state.round_result = f"You win {int(abs(result))} chip(s)!"
            st.session_state.player_score += int(abs(result))
        elif result < 0:
            st.session_state.round_result = f"Bot wins {int(abs(result))} chip(s)!"
            st.session_state.bot_score += int(abs(result))
        else:
            st.session_state.round_result = "It's a tie!"

    # UI Layout
    col_score1, col_score2 = st.columns(2)
    col_score1.metric("Your Score", st.session_state.player_score)
    col_score2.metric("Bot Score", st.session_state.bot_score)

    st.divider()

    if not st.session_state.game_active and not st.session_state.waiting_for_new_hand:
        if st.button("Deal New Hand", type="primary"):
            start_new_hand()
            st.rerun()
    elif st.session_state.waiting_for_new_hand:
        st.markdown(f"### {st.session_state.round_result}")
        st.markdown(
            f"Your card: **{CARD_NAMES[st.session_state.player_card]}** | "
            f"Bot's card: **{CARD_NAMES[st.session_state.bot_card]}**"
        )
        if st.button("Deal New Hand", type="primary"):
            start_new_hand()
            st.rerun()
    else:
        # Show player's card
        st.markdown(f"### Your card: **{CARD_NAMES[st.session_state.player_card]}**")
        position = "first" if st.session_state.player_is_p0 else "second"
        st.markdown(f"You are acting **{position}**.")

        # Show action history
        if st.session_state.game_log:
            st.markdown("**Action history:**")
            for log_entry in st.session_state.game_log[-5:]:
                st.markdown(f"- {log_entry}")

        # Action buttons
        st.markdown("---")
        st.markdown("**Your turn:**")
        col_a, col_b = st.columns(2)
        h = tuple(st.session_state.game_history)

        # Determine if we're facing a bet
        facing_bet = len(h) > 0 and h[-1] == BET

        with col_a:
            label = "Fold" if facing_bet else "Check"
            if st.button(label, use_container_width=True):
                player_act(PASS)
                st.rerun()
        with col_b:
            label = "Call" if facing_bet else "Bet"
            if st.button(label, use_container_width=True, type="primary"):
                player_act(BET)
                st.rerun()
