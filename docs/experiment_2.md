# Experiment 2: LSTM Baseline Reproduction

**Project:** ShengJi++ — Deep RL AI for Tractor (Shengji)
**Date:** 2026-04-01
**Infrastructure:** Vast.ai — AMD EPYC 9554 (Zen 4), RTX 4090, 20.5 allocated cores, $0.33/hr

---

## Overview

This experiment reproduced the original LSTM-based architecture from the paper to establish a baseline. Training was cut short at 196K games (vs. 700K in the paper) but already showed the LSTM outperforming the small transformer from Experiment 1, revealing that model capacity is the key bottleneck.

---

## 1. Training Command

```bash
OMP_NUM_THREADS=1 python TrainLoop.py \
  --model-folder exps/lstm_baseline \
  --games 2000 \
  --max-games 700000 \
  --eval-size 300 \
  --enable-combos \
  --combo-penalty 0.01 \
  --discount 0.95 \
  --epsilon 0.015 \
  --actor-processes 18 \
  --eval-processes 6
```

Note: no `--model-architecture transformer` flag — defaults to the original LSTM (`networks/Models.py`).

---

## 2. Model Architecture Comparison

The transformer only replaces the **MainModel** — Declaration, Kitty, and Chaodi models are identical FC networks across all experiments.

### MainModel Parameter Counts

| Architecture | d_model / hidden | Layers | ~Params (Main only) | vs. LSTM |
|---|---|---|---|---|
| LSTM (original) | hidden=256 | 1 LSTM + 5 FC | ~2,760K | — |
| Transformer v1 (Exp 1) | d_model=64 | 2 | ~124K | 22× smaller |
| **Transformer v2 (next)** | **d_model=128** | **4** | **~860K** | **3× smaller** |

### Shared Model Parameter Counts (same across all experiments)

| Model | Architecture | ~Params |
|---|---|---|
| DeclarationModel | 3-layer FC (179→256→256→1) | ~112K |
| KittyModel | 4-layer FC + embedding (226→256→256→256→1) | ~193K |
| ChaodiModel | 4-layer FC (178→256→256→256→1) | ~178K |
| **Shared total** | | **~483K** |

### Full Suite Totals

| Experiment | MainModel | Shared | Total |
|---|---|---|---|
| LSTM baseline | ~2,760K | ~483K | **~3,243K** |
| Transformer Exp 1 | ~124K | ~483K | **~607K** |
| Transformer Exp 2 (planned) | ~860K | ~483K | **~1,343K** |

The LSTM MainModel is large primarily due to its deep FC head (768→512→512→512→512→1) after the LSTM layer. The transformer achieves comparable expressiveness with a much smaller parameter count by using attention over the history sequence instead of compressing it to a single hidden state.

---

## 3. Results

### Training Speed

```
Training time: 224.8s (8.9 games/s)  [final checkpoint]
  Queue wait: 205.9s (92%)  Learn: 18.9s (8%)
  Avg sim/game: 1.77s  Avg inference/game: 0.10s (6% of sim)
```

The LSTM is faster per inference step than the transformer (0.10s vs 0.22s) since it processes history as a single LSTM pass rather than full attention over the sequence.

### Evaluation vs. Random Agent (196K games checkpoint)

```
Win counts:   [269, 31]
Win rate:     89.7%
Level rate:   94.3%
Avg opp pts: 143.4 vs 56.3
```

### Leveling Rate Comparison (vs. Transformer Exp 1)

| Games | LSTM Level Rate | Transformer Level Rate |
|---|---|---|
| 2K | 84.2% | 80.4% |
| 50K | ~91% | ~88% |
| 100K | ~93% | ~90% |
| 196K | **94.3%** | — |
| 300K | — | 92.2% |

The LSTM reached 94.3% at 196K games. The transformer only reached 92.2% at 300K games — with 50% more training and still behind.

---

## 4. Analysis

### Why LSTM Outperforms the Small Transformer

The LSTM MainModel has ~2.76M parameters vs. ~124K for the transformer — a **22× difference in capacity**. The LSTM's large FC head (5 layers of 512 units after the LSTM) gives it significantly more representational power for evaluating action quality.

The transformer architecture is theoretically better suited to sequential history (attention vs. fixed hidden state), but the d_model=64, 2-layer configuration was too small to demonstrate this advantage. Model capacity dominated architectural choice at this scale.

### LSTM Inference Is Faster

The LSTM processes history in a single sequential pass and compresses it to a 256-dim hidden state. The transformer applies full self-attention over 15 history tokens, which is slower but more expressive at larger scales. At d_model=64 the transformer's theoretical advantage doesn't materialize.

---

## 5. Next Steps

### Experiment 3: Larger Transformer

Restore the transformer to d_model=128, nhead=8, num_layers=4, dim_feedforward=512 (~860K params). This brings the transformer to roughly 1/3 of the LSTM's parameter count while retaining the architectural advantages of attention over history.

OOM was the original reason for reducing model size. With the fixes from Experiment 1 (chunked inference + pre-tensorization), the larger model should fit within VRAM bounds.

Expected training command:
```bash
OMP_NUM_THREADS=1 python TrainLoop.py \
  --model-folder exps/transformer_v3 \
  --model-architecture transformer \
  --games 2000 \
  --max-games 700000 \
  --eval-size 300 \
  --enable-combos \
  --combo-penalty 0.01 \
  --discount 0.95 \
  --epsilon 0.015 \
  --actor-processes 18 \
  --eval-processes 6
```

Before running: update `networks/TransformerModels.py` to restore d_model=128, nhead=8, num_layers=4, dim_feedforward=512.
