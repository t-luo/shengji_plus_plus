# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ShengJi++ is a deep reinforcement learning AI for the Tractor card game (Shengji), a 4-player trick-taking game with 2 decks. Developed as a Master's thesis project. Only external dependency is PyTorch.

## Commands

**Run tests:**
```bash
python -m unittest discover -s tests -v
```

**Train a model:**
```bash
python TrainLoop.py --model-folder <SAVE_PATH> --discount 0.95 --epsilon 0.015 --games 2000 --eval-size 300 --enable-combos --combo-penalty 0.01
```

**Evaluate a model:**
```bash
# vs. random baseline
python TrainLoop.py --model-folder <SAVE_PATH> --eval-only --eval-size 3000

# vs. rule-based baseline
python TrainLoop.py --model-folder <SAVE_PATH> --eval-only --eval-agent strategic --eval-size 1000

# compare two models
python TrainLoop.py --eval-only --model-folder model1 --compare model2 --eval-size 1000
```

**Play interactively against the AI:**
```bash
python TrainLoop.py --eval-only --model-folder <SAVE_PATH> --eval-agent interactive --single-process --eval-size 1 --verbose
```

## Architecture

### Game Stages
The game has four sequential stages, each with a dedicated neural network and agent module:
1. **Declaration** – Players draw cards and declare a trump suit
2. **Kitty** – Dealer discards 8 cards into the kitty (hidden cache)
3. **Chaodi** (optional) – A player may swap trump and exchange with the kitty
4. **Main** – Trick-taking phase; teams earn points by capturing point cards

### Agent Structure (`agents/`)
- `Agent.py` – `SJAgent` base class that composes four `StageModule` instances (one per stage)
- `DMCAgent.py` / `DQNAgent.py` – RL algorithm implementations
- `RLAgents.py` – Concrete stage modules (`DeclareAgent`, `KittyAgent`, `ChaodiAgent`, `MainAgent`) that wrap the networks
- `StrategicAgent.py` – Rule-based heuristic baseline
- `InteractiveAgent.py` – Human input agent for interactive play

### Environment (`env/`)
- `Game.py` – Game state machine; enforces rules, generates valid actions, computes observations
- `CardSet.py` – Card representation, move validation, tractor/combo detection (central data structure)
- `Actions.py` – Enumerated action types for each stage
- `Observation.py` – Per-player observable game state

### Networks (`networks/`)
- `Models.py` – Stage-specific networks: `DeclarationModel`, `KittyModel`, `ChaodiModel`, `MainModel`, `ValueModel` (SAC)
- `StateAutoEncoder.py` – Autoencoder for compact state representation

### Training Pipeline
`TrainLoop.py` orchestrates multi-process training using PyTorch's `spawn` context:
- **Sampler processes** – Run self-play games via `Simulation.py` and push experiences into shared queues
- **Evaluator processes** – Benchmark the current model against a baseline agent
- Separate replay buffers per game stage; supports DQN, DMC, and SAC algorithms

`Simulation.py` wraps `Game.py`, manages per-stage reward histories, and returns training trajectories.

### Team Structure
Players 0 & 2 are on one team; players 1 & 3 are on the other. The "Level" team tries to reach a target level; the "Attacking" team tries to prevent this by capturing enough point cards.
