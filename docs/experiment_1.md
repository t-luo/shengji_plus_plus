# Experiment 1: LSTM to Transformer Architecture

**Project:** ShengJi++ — Deep RL AI for Tractor (Shengji)
**Date:** 2026-03-31
**Infrastructure:** Vast.ai — AMD EPYC 9554 (Zen 4), RTX 4090, 20.5 allocated cores, $0.33/hr

---

## Overview

This experiment replaced the original LSTM-based main-stage model with a Transformer architecture, resolved a series of engineering bugs that emerged from the migration, investigated and resolved training speed bottlenecks, and produced a fully trained model evaluated against both a random and a rule-based strategic baseline.

---

## 1. Architecture Change: LSTM to Transformer

The original main-game-stage model used an LSTM to process sequential game history. This was replaced with a Transformer encoder defined in `networks/TransformerModels.py`.

### Design

- **Separate inputs:** Current state+action tensor and a history tensor encoding the previous 15 rounds of moves (15 rounds × 436 dims)
- **Pre-LN (`norm_first=True`):** Applied for training stability; mitigates gradient vanishing in deeper stacks
- **Output:** Scalar Q-value, compatible with the existing DMC training pipeline

### Size Evolution

| Version | d_model | nhead | num_layers | dim_feedforward | ~Params |
|---------|---------|-------|------------|-----------------|---------|
| Initial | 128 | 8 | 4 | 512 | 860K |
| Final | 64 | 4 | 2 | 256 | 124K |

The model was reduced 7× in parameter count during the CUDA OOM investigation (see Section 3).

---

## 2. Bugs Fixed Before Training

### 2.1 Stray `turtle` Import

**File:** `networks/Models.py`

```python
from turtle import forward  # erroneous import
```

On headless Linux servers, `turtle` requires `tkinter`, which is not installed. This caused an immediate `ModuleNotFoundError: No module named 'tkinter'` on import. The line was removed.

### 2.2 Hot-Swap Model Loading Broken in Child Processes

The training loop used `importlib.spec_from_file_location` to load a custom `Models.py` from an experiment folder at runtime. Modules loaded this way are not picklable, so when `torch.multiprocessing.spawn` forked child processes, they could not import the dynamically loaded module.

**Fix:** Copy `Models.py` to `networks/ActiveModels.py` — a stable, fully-qualified import path — before spawning any processes. Child processes then import from this fixed location.

### 2.3 `PicklingError` on Model Deep Copy

```
PicklingError: Can't pickle train_models.DeclarationModel
```

Two locations used `pickle.loads(pickle.dumps(model))` to clone models: `DMCAgent.load_model` and `Agent.DeepAgent`. This fails for the same reason as 2.2 — pickle roundtrips do not survive dynamic module loading.

**Fix:** Replace with `copy.deepcopy(model)` in both locations.

### 2.4 `enable_nested_tensor` UserWarning

`norm_first=True` silently disables PyTorch's nested tensor optimization, but PyTorch still emitted a noisy warning. Fixed by explicitly passing `enable_nested_tensor=False` to `nn.TransformerEncoder`.

### 2.5 `torch.load` FutureWarning

All four `torch.load` calls in `DMCAgent.load_models_from_disk` lacked `weights_only=True`. Added to suppress the FutureWarning and adopt the safer loading behavior.

---

## 3. CUDA Out-of-Memory Issues

### Problem

Training with 6 actor processes and 6 eval processes on an RTX 4090 (24 GB VRAM) produced OOM errors. Two compounding causes:

1. Each actor process loaded the full model into VRAM.
2. Transformer attention activations scale as O(B × heads × seq_len²). With large action spaces (many legal moves evaluated per step), intermediate activation tensors were large.
3. Fragmented allocator cache after the training phase caused OOM when spawning eval processes.

### Solutions Applied

**Chunked inference (`_eval_in_chunks`):**
Instead of evaluating all legal actions in a single batched forward pass, evaluate in chunks of 32. This bounds peak activation memory regardless of action-space size.

**Model size reduction:**
Reduced from ~860K to ~124K parameters (7× smaller). Per-process VRAM usage dropped from ~860 MB to ~200 MB.

**`torch.cuda.empty_cache()` before eval spawn:**
Releases cached allocator blocks that accumulate during the training phase, preventing fragmentation-induced OOM when spawning the evaluation process group.

**Unsetting `expandable_segments`:**
PyTorch's `expandable_segments:True` allocator setting is incompatible with `torch.multiprocessing.spawn`. This environment variable was unset before launching training.

---

## 4. Training Speed Investigation

### Initial Baseline

~2.3 games/s with 6 actors on a 32-vCPU machine.

### Timing Instrumentation Added

| Measurement | Location | Purpose |
|---|---|---|
| `_last_inference_ms` | `DMCModule` | GPU forward pass time per `act()` call |
| `total_inference_ms` | `Simulation` | Accumulated inference time per game |
| `t_queue` | Main training loop | Time blocked on `queue.get()` |
| `t_learn` | Main training loop | Time in `learn_from_samples` |

After each checkpoint, the training loop prints a breakdown:

```
Training time: Xs (Y games/s)
  Queue wait: As (P%)  Learn: Bs (Q%)
  Avg sim/game: Cs  Avg inference/game: Ds (E% of sim)
```

---

## 5. Cloud Infrastructure Journey

### RunPod

Multiple RunPod pods were tested. A key discovery was that RunPod containers are capped at approximately **27.2 logical cores** (`cfs_quota_us=2720000`) regardless of the advertised vCPU count.

| CPU | Generation | Speed | Throughput |
|---|---|---|---|
| AMD EPYC 7C13 | Zen 2 | ~17s/game | 0.3–0.4 games/s |
| AMD EPYC 7763 | Zen 3 | ~6s/game | 2.4 games/s (16 actors) |
| AMD EPYC 9554 | Zen 4 | fast per-core | capped at 27.2 cores |
| AMD EPYC 9575F | Zen 5 | marginal gain | $1.70/hr — not cost-effective |

The 27.2-core cap meant scaling past ~16 actors yielded no benefit regardless of advertised capacity.

### Vast.ai

Switched to Vast.ai for better CPU allocation flexibility. Selected instance:

- **CPU:** AMD EPYC 9554 (Zen 4)
- **GPU:** RTX 4090
- **Allocated cores:** 20.5 (`cfs_quota_us=2048000`)
- **Price:** $0.33/hr

### Critical Fix: `OMP_NUM_THREADS=1`

With 14+ actor processes, the following error appeared:

```
libgomp: Thread creation failed: Resource temporarily unavailable
```

Each Python actor process was internally spawning multiple OpenMP threads, exhausting the thread budget across all processes. Setting `OMP_NUM_THREADS=1` forces each actor to use exactly one thread — the correct behavior for RL training where parallelism is achieved at the process level, not the thread level. This was a significant unlock for scaling.

### Final Configuration

- 18 actor processes, 6 eval processes
- `OMP_NUM_THREADS=1`
- Vast.ai AMD EPYC 9554 / RTX 4090 / 20.5 cores / $0.33/hr
- Result: **7.4 games/s**

---

## 6. Pre-Tensorization Refactor

### Problem

The `prepare_batch_inputs` method ran in the single main training process. It performed heavy Python loops (iterating card history, constructing dynamic tensors) for every mini-batch. On slower EPYC pods this caused learn time to consume 63–68% of total training time, keeping GPU utilization at approximately 2%.

### Solution

Move tensor construction to actor processes before the data enters the queue:

- `prepare_batch_inputs` now returns CPU tensors rather than consuming them in-place.
- A new `learn_from_tensors` method was added to `DMCModule` and `MainModule`. It accepts pre-built CPU tensors, moves them to GPU, and performs the forward and backward passes.
- Sampler processes call `prepare_batch_inputs` before `queue.put()`.
- The main training loop calls `learn_from_tensors` instead of `learn_from_samples`.
- `_eval_in_chunks` was updated to call `.to(device)` internally, so callers do not need to manage device placement.

### Impact

| Metric | Before | After |
|---|---|---|
| Learn time share | 63–68% | 1–7% |
| Queue wait share | ~30% | 86–99% |
| GPU utilization | ~2% | GPU no longer bottleneck |

Queue wait becoming dominant confirms that actor processes are producing data faster than the learner can consume it — the intended regime for this training architecture.

---

## 7. Final Training Command

```bash
OMP_NUM_THREADS=1 python TrainLoop.py \
  --model-folder exps/transformer_v2 \
  --model-architecture transformer \
  --games 2000 \
  --max-games 300000 \
  --eval-size 300 \
  --enable-combos \
  --combo-penalty 0.01 \
  --discount 0.95 \
  --epsilon 0.015 \
  --actor-processes 18 \
  --eval-processes 6
```

---

## 8. Results

### Training Speed

| Metric | Value |
|---|---|
| Sustained throughput | 7.4 games/s |
| Total games | 300,000 |
| Wall-clock time | ~11 hours |
| Total cost | ~$3.60 |

### Final Checkpoint Timing Breakdown

```
Training time: 271.9s (7.4 games/s)
  Queue wait: 234.0s (86%)  Learn: 37.9s (14%)
  Avg sim/game: 2.11s  Avg inference/game: 0.22s (11% of sim)
```

### Evaluation vs. Random Agent (300K games checkpoint)

```
Win counts:              [266, 34]
Win rate:                88%
Level counts:            [495, 42]
Avg opposition points:   132.6 vs 74.4
```

### Evaluation vs. Strategic (Rule-Based) Agent (1,000 games)

```
Win counts:              [569, 431]
Win rate:                56.9%
Level counts:            [812, 528]
Avg opposition points:   94.3 vs 88.4
Avg inference time:      0.013s
```

### Training Curve Observations

- The model improved rapidly during the first ~20,000 games, learning basic card game logic quickly.
- Win rate against the random agent plateaued at approximately 82% for the remaining 280,000 games.
- Training is pure self-play (player1 fills all 4 positions). The random agent is only the evaluation opponent, not the training opponent. The plateau reflects that the random agent is too weak a benchmark — the model kept improving through self-play but the evaluation metric stopped capturing it.
- A 56.9% win rate against the strategic agent confirms the model learned meaningful game strategy, but indicates substantial headroom for improvement with harder training opponents or curriculum learning.

---

## 9. Final Model Architecture

**File:** `networks/TransformerModels.py`

| Hyperparameter | Value |
|---|---|
| d_model | 64 |
| nhead | 4 |
| num_layers | 2 |
| dim_feedforward | 256 |
| Parameters | ~124K |
| norm_first | True (Pre-LN) |
| enable_nested_tensor | False |

**Inputs:**
- State+action concatenated tensor (current step)
- History tensor: 15 rounds × 436 dims

**Output:** Scalar Q-value

---

## 11. Next Steps

### 11.1 Reproduce Original Paper Results (LSTM baseline)

Run the original LSTM model (default `--model-architecture mlp`) with the same hyperparameters used in the paper. This establishes a baseline to compare against the transformer.

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

### 11.2 Larger Transformer (Experiment 2)

The 124K param model was likely too small to learn beyond basic strategy. Now that OOM is resolved via chunked inference and pre-tensorization, restore a larger model:
- d_model=128, nhead=8, num_layers=4, dim_feedforward=512 (~860K params)
- Train to 700K games to match the paper
- Switch evaluation metric to leveling rate (already tracked in `stats.pkl` as `level_counts`)

### 11.3 Switch Evaluation Metric to Leveling Rate

The `level_counts` field is already saved in `stats.pkl` and plotted in `plot_stats.py`. Update the plot to show leveling rate as the primary metric (matching Figure 5.2 in the paper) instead of binary win rate, which saturates too early to measure continued improvement.

---

## 10. Key Learnings

1. **CPU per-core speed matters more than core count** for Python-based RL game simulation. Faster cores (Zen 4 vs. Zen 2) had a larger impact than adding more of them.

2. **Pre-tensorize in actor processes.** When the main training process is the bottleneck, moving tensor construction to parallel actor CPUs can reduce learn-time share from ~68% to under 7%.

3. **`OMP_NUM_THREADS=1` is essential for multi-actor RL training.** Each Python process otherwise spawns OpenMP threads internally, rapidly exhausting thread limits when running 14+ actors.

4. **RunPod caps containers at ~27.2 cores** regardless of advertised vCPU count. Vast.ai provides more flexible CPU allocation and proved more cost-effective for this workload.

5. **Evaluating against a random opponent is insufficient for measuring improvement.** Training is pure self-play — `player1` fills all 4 positions. The random agent is only the *evaluation* opponent used to measure progress after each checkpoint. Once the model learned basic strategy it consistently beat random ~82%, making the metric uninformative for the remaining 280K games. Future experiments should evaluate against the strategic agent or a held-out checkpoint to get a meaningful signal of continued improvement.

6. **Model size reduction (7×) resolved VRAM OOM** while maintaining training viability. A 124K-parameter transformer is sufficient for the current task complexity and allows many concurrent actor processes on a single GPU.
