# Where We Went Wrong, and How to Beat the LSTM

**Date:** 2026-09-11
**Companion to:** [`summary.md`](summary.md), [`experiment_1.md`](experiment_1.md), [`experiment_2.md`](experiment_2.md)
**Goal:** get the attention-based main-stage model past the LSTM baseline (94.3% level rate
@196K games) and past the original paper (97.7% @1.2M games).

> **Written to be read cold.** §1 is a refresher on how the system works — skip it if the
> project is fresh in your mind. §2 is the headline diagnosis. §3–4 are what went wrong.
> §5–6 are what to do about it.

---

## Status as of 2026-11-12

Everything below this line is the original diagnosis, unmodified, for the historical record.
The E0 instrumentation/hygiene pass and most of E1's transformer-specific fixes have since
landed. See [`experiment_4_runbook.md`](experiment_4_runbook.md) for the exact commands to run
next and what's still explicitly deferred.

| Item | Status | Where |
|---|---|---|
| §4.1 NaN resets weights mid-run | **Fixed** | `DMCModule._step_optimizer` skips the batch, counts it, raises past a threshold |
| §4.2 optimizer-load failure is a warning | **Fixed** | `TrainLoop.train()` raises unless `--allow-fresh-optimizer` |
| §4.3a main-stage ε-greedy no-op | **Fixed** | `MainModule.act()` |
| §4.3b ε resets on resume / scales with actor count | **Fixed** | `TrainLoop.epsilon_at()`, pure function of global game count |
| §4.3b "lower the other three stages' ε" (the sweep itself) | **Not done** | still one shared ε across all 4 stages; flagged as a follow-up ablation |
| §4.4 history drops the leader one-hot | **Fixed** | `_TransformerBase._enc_history` |
| §4.4 "ideally" trick-winner flag + point value | **Not done** | would need CardSet trick-resolution logic inside `forward()`; deferred |
| §4.5 Pre-LN with no final norm | **Fixed** | `norm=nn.LayerNorm(d_model)` |
| §4.6 history padding never masked | **Fixed** | `_TransformerBase._forward_impl` |
| §4.7 cardset tokens are purely linear | **Fixed** | `cardset_mlp`/`hand_mlp` |
| §4.8a accidental entropy bonus at α=1.0 always on | **Fixed** | gated on `self.sac` in both `learn_from_*` |
| §4.8b entropy uses mean instead of sum | **Fixed** | `MainModule.act()` |
| §4.9 points tensor duplicated / defenders invisible | **Fixed** | one-word fix in `Game.get_observation` |
| §4.10 grad clip at 80 is a no-op | **Fixed** | default tightened to 1.0, `--grad-clip` to override |
| §4.10 `strict=False` on load | **Fixed** | `strict=True` everywhere in `DMCAgent.load_models_from_disk` |
| §4.10 LR hardcoded, no schedule/warmup | **Fixed** | `--lr`/`--optimizer`/`--weight-decay`/`--warmup-steps` (warmup not persisted across resumes — documented limitation) |
| §4.10 `tau=0.1` soft update runs once per queue item | **Not done** | unchanged; still couples effective algorithm to actor count |
| §2/E2 Q-head capacity (highest predicted value) | **Fixed** | ~1.3M-param head reading raw state + CLS |
| §3.2 eval budget divided across opponents | **Fixed** | `--eval-size` is now per-opponent; Wilson 95% CI logged |
| E0 item 3: no training diagnostics persisted | **Fixed** | `stats.pkl` now carries loss/grad-norm percentiles, `nan_batches`, ε per round |
| E0 item 6: sync command.txt/state.pkl/snapshots back from cloud | **Operational, not code** | added to the runbook's checklist |
| E4.4 Boltzmann sampling over Q-values | **Not done** | exists unused in the 2022 codebase; not wired up this round |
| DQNAgent.py (non-DMC path) | **Untouched** | out of scope — CLAUDE.md's documented path is DMC |

---

## Contents

1. [Refresher: how this system actually works](#1-refresher-how-this-system-actually-works)
2. [The headline diagnosis: we scaled the wrong half of the network](#2-the-headline-diagnosis-we-scaled-the-wrong-half-of-the-network)
3. [Methodology problems — why our conclusions don't hold](#3-methodology-problems--why-our-conclusions-dont-hold)
4. [Code defects found by reading the training path](#4-code-defects-found-by-reading-the-training-path)
5. [The throughput bug that unlocks everything else](#5-the-throughput-bug-that-unlocks-everything-else)
6. [How to actually beat the LSTM](#6-how-to-actually-beat-the-lstm)
7. [Suggested order of work](#7-suggested-order-of-work)
8. [Confidence levels](#8-confidence-levels)

---

## 1. Refresher: how this system actually works

### 1.1 The game and the four stages

Tractor/Shengji is a 4-player, 2-deck, trick-taking game. North+South are one team,
East+West the other. One team are the **defenders** (the declarer's team); the other are the
**opponents**, who try to capture ≥80 points. Point cards are 5s (5 pts), 10s and Kings
(10 pts each) — **200 points total in the two decks**.

A game runs through four stages, and **each stage has its own neural network and its own
replay queue**:

| Stage | What happens | Network | Typical decisions/game |
|---|---|---|---|
| 1. Declaration | Players draw and may declare a trump suit | `DeclarationModel` (3-layer FC) | a few |
| 2. Kitty | Dealer buries 8 cards | `KittyModel` (4-layer FC) | **8 sequential** |
| 3. Chaodi (optional) | A player may override the trump declaration | `ChaodiModel` (4-layer FC) | 0–few |
| 4. **Main** | Trick-taking; this is where the game is won | **`MainModel`** ← all our work | ~25 per player |

Only `MainModel` was ever changed by Experiments 1–3. The other three networks are identical
FC nets across every experiment (~483K params total) and were never touched.

### 1.2 The learning algorithm (DMC)

Deep Monte Carlo, not Q-learning. There is **no bootstrapping and no target network** in the
usual sense:

1. Play a full game to completion with the current policy (self-play, all 4 seats).
2. When the game ends, walk each player's trajectory **backwards** and compute a discounted
   return-to-go ([Simulation.py:223-254](Simulation.py#L223-L254)):
   `rw_i = (points won/lost this trick, scaled) + 0.95 · rw_{i+1}`, with the final step
   also receiving the terminal reward.
3. The terminal reward is the **level change**: `+1` if opponents reach 80 pts (`1 + (pts-80)//40`
   for bigger wins), `-1` if they get 40–79, `-2` if 1–39, `-3` if shut out
   ([Game.py:383-394](env/Game.py#L383-L394)). Defenders get the negation.
4. `MainModel` is then trained by **plain MSE regression** of `Q(state, action)` onto that
   return. That's it — the "Q-function" is a supervised regressor on Monte Carlo returns.
5. To act, the agent enumerates **every legal action**, scores each one with a forward pass,
   and takes the argmax ([DMCAgent.py:259-276](agents/DMCAgent.py#L259-L276)).

Return magnitudes end up roughly in the **±1 to ±5** range. Remember that number — several
bugs below are best understood as a fraction of it.

### 1.3 The training pipeline

`TrainLoop.py` runs an actor–learner setup over `torch.multiprocessing` with `spawn`:

- **18 actor processes** (`sampler`, [TrainLoop.py:30-78](TrainLoop.py#L30-L78)) play self-play
  games in batches of 10, convert trajectories to tensors, and push them onto four shared
  queues (one per stage).
- **1 main process** pops from the queues and does GPU forward/backward
  ([TrainLoop.py:232-246](TrainLoop.py#L232-L246)).
- Every `--games` (2000) games it checkpoints, then spawns **6 evaluator processes** to
  benchmark against opponents, then appends a row to `stats.pkl`.
- Each stage module holds **two copies** of its network: `_model` (trained) and `_eval_model`
  (used by actors to act, in shared memory, updated by a soft copy `tau=0.1` after each batch).

**Crucial property:** queue wait is **86–92%** of wall-clock time in every run. The learner is
starved; the actors are the bottleneck. **Sampling throughput *is* the experiment budget.**

### 1.4 What the metrics mean

- **Win rate** — fraction of evaluation games our side won. Saturates early against weak
  opponents, which is why Experiment 1 recommended abandoning it.
- **Level rate** — `level_counts[us] / (level_counts[us] + level_counts[them])`, where each game
  contributes `|terminal reward|` (1, 2, 3…) to the winner. It is a **magnitude-weighted win
  rate**: beating an opponent 3 levels counts triple. More sensitive, hence preferred.
- **Avg opposition points** — mean points the attacking side scored, split by which team dealt.
  Lower when we defend, higher when we attack, is better.
- **Opponents**: `random` ([RandomAgent.py](agents/RandomAgent.py), uniform over legal moves
  with a mild anti-combo bias), `strategic` ([StrategicAgent.py](agents/StrategicAgent.py),
  hand-written heuristics — e.g. declare only with >5 trumps covering ≥40% of the hand), and
  `model:<folder>` (a frozen checkpoint, added in Experiment 3).

### 1.5 The experiment lineage, in one table

| Run | Architecture | Games | Eval opponents | Outcome |
|---|---|---|---|---|
| `transformer_v2` | Transformer d=64, 2L (**124K**) | 300K | random | 92.2% level rate |
| `lstm_baseline` | Original LSTM (**2.76M**) | 196K (cut short) | random | **94.3%** — beat the transformer with 35% less training |
| `transformer_v3` | Transformer d=128, 4L (**860K**) | 700K | random | Peaked ~95% @360K, then **decayed to 82.6%** |
| `transformer_v3_b` | same | 318K | random + strategic + lstm | Improved steadily throughout |
| `transformer_v3_c` | same | 482K | random + strategic + lstm | Tracked v3_b, then **dropped hard at ~320K** and never recovered |

The conclusion recorded in `experiment_2.md` was *"model capacity dominated architectural
choice"* — and that conclusion drove the Experiment 3 decision to scale d_model 64→128 and
2→4 layers. **That conclusion was wrong**, and §2 explains why.

---

## 2. The headline diagnosis: we scaled the wrong half of the network

Both networks split cleanly into two halves: an **encoder** that turns the raw game state into
features, and a **Q-head** that turns those features into a single scalar value. Here is where
the parameters actually sit:

| | Sequence/state encoder | **Q-head** | Total |
|---|---|---|---|
| LSTM `MainModel` | 710,656 | **2,050,049** | 2,760,705 |
| Transformer Exp 3 (d=128, 4L) | 793,088 | **8,321** | 856,577 |
| Transformer Exp 1 (d=64, 2L) | 99,968 | **4,225** | 123,585 |

*(Analytic counts. They reproduce the 124K / 860K / 2.76M totals in `experiment_2.md` exactly,
so the accounting is trustworthy.)*

Read the middle column. At Experiment 3 scale the transformer's encoder is **12% larger** than
the LSTM's recurrent encoder — the "capacity" problem in the part we were actually theorising
about was *already solved*. What remained 246× smaller is the head:

```python
# networks/TransformerModels.py:203-207  — the ENTIRE Q-head, 8,321 params
self.head = nn.Sequential(
    nn.Linear(d_model, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
```

versus the LSTM's ([Models.py:166-178](networks/Models.py#L166-L178)):

```python
self.fc1 = nn.Linear(1129, 768)     # 867K
self.fc2 = nn.Linear(768, 512)      # 393K
self.fc_rest = nn.Sequential(       # 786K  (3 × 512→512 with ReLU)
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU())
self.fc_final = nn.Linear(512, 1)
```

### Why this matters so much for *this* algorithm

Recall from §1.2 that DMC is **pure regression onto Monte Carlo returns**, and that acting
means **ranking every legal action by its scalar score**. The head is the only part of the
network that performs that ranking. Everything the encoder produces has to be funnelled through
a 128→64→1 bottleneck before any action can be compared to any other.

There's a second, subtler asymmetry. The LSTM's head receives `hstack([x, history_out])` —
the **raw, uncompressed 873-dim state vector** concatenated with the LSTM summary. Every card
indicator reaches the head directly. The transformer's head sees **only the 128-dim CLS
token**: the entire game state has been compressed ~7× before the head ever sees it, and the
head then compresses it another 2× before scoring.

So the comparison we ran was never "LSTM vs Transformer." It was **"2.05M-param head on raw
features" vs "8.3K-param head on a 128-dim bottleneck"** — plus five other confounds (§3.1).

---

## 3. Methodology problems — why our conclusions don't hold

### 3.1 The architecture comparison was confounded on at least six axes

Experiments 1–3 changed all of the following simultaneously, then attributed the result to
"LSTM vs Transformer":

| Axis | LSTM baseline | Transformer | Why it matters |
|---|---|---|---|
| History encoder | LSTM over 15×436 | Self-attention over 60 tokens | The thing we *meant* to test |
| **Q-head capacity** | **2.05M params** | **8.3K params** | §2 — probably the whole story |
| Head input | Raw 873-dim state + LSTM summary | 128-dim CLS only | 7× information bottleneck |
| Per-cardset encoding | Raw 108-dim vector → ReLU MLP | **Purely linear** `count @ W` → `Linear` | §4.7 — no nonlinearity at all |
| Dropout | **0.0** | **0.1** in every encoder layer | Unmatched regularisation |
| Optimizer | RMSprop lr=1e-4 (tuned for it over years) | RMSprop lr=1e-4 (**inherited, never tuned**) | Transformers usually want AdamW + warmup |

Any one of rows 2–6 could account for the entire observed gap. **No single-variable comparison
exists anywhere in the project.** This is the central methodological failure: we ran a
6-variable change and drew a 1-variable conclusion, then spent Experiment 3 acting on it.

### 3.2 The evaluation budget was split until it became noise

[TrainLoop.py:260](TrainLoop.py#L260):

```python
games_per_eval = max(1, eval_size // len(eval_agent_list))
```

`--eval-size 300` with three opponents gives **100 games each** — confirmed in the raw stats
(`win_counts: [43, 57]` sums to 100). At p≈0.5 the binomial standard error over 100 games is
**±5 percentage points**, so a single checkpoint carries a **±10pp 95% confidence interval**.

Look again at the `v3_c` curve with that in mind: most of the visible sawtooth is sampling
noise, and we were reading trends off it. (The ~15pp sustained drop at 320K is larger than the
noise band and *is* real — but individual checkpoint-to-checkpoint wiggles are not.)

Worse, the top-line `win_counts` / `level_counts` fields in `stats.pkl`
([TrainLoop.py:320-328](TrainLoop.py#L320-L328)) are computed by **summing across opponents of
wildly different strength**:

```python
for s in per_opponent_stats.values():
    all_win[0] += s['win_counts'][0]; all_win[1] += s['win_counts'][1]
```

That number is not a coherent quantity — it moves when the *opponent mix* changes, not only
when the model does. Concretely: **`v3`'s top-line (random only) and `v3_b`/`v3_c`'s top-line
(random + strategic + lstm) are not comparable at all**, yet they are plotted on the same axes
in `curve_vs_all.png`. `v3_b`/`v3_c` look catastrophically worse than `v2`/`lstm_baseline`
largely because two-thirds of their eval games are against much stronger opponents.

### 3.3 No training diagnostics were ever persisted

Loss is printed once per checkpoint and then **deleted** ([TrainLoop.py:252](TrainLoop.py#L252),
then `clear_loss_histories()` at [:257](TrainLoop.py#L257)). `stats.pkl` records only eval
outcomes and timings. There is **no record of** loss, gradient norms, NaN events, epsilon, or
queue lag.

This is why the `v3` decay and the `v3_c` step-change are unexplainable today. As §4.1 shows,
the most likely culprit is an event that leaves *no trace whatsoever* in anything we kept.

### 3.4 Single runs, no seed replication

Every configuration was run exactly once. With ±5pp eval noise and a stochastic self-play
pipeline, we cannot distinguish a real regression from seed variance.

`v3_b` and `v3_c` are the closest thing we have to a seed replicate — and they diverge by
10–15 points after 320K games. **That divergence is arguably the most important datapoint in
the whole project**: it says run-to-run variance is large enough to swamp most of the effects
we were trying to measure. Every conclusion drawn from a single run is therefore provisional.

### 3.5 Provenance was lost

`TrainLoop.py` *does* write `command.txt` ([TrainLoop.py:196-198](TrainLoop.py#L196-L198)) —
but only `stats.pkl` and the PNGs were synced back from the Vast.ai boxes. The `exps/transformer_v3*`
folders contain no `command.txt`, no `state.pkl`, no weights. We cannot now determine what
distinguished `v3_b` from `v3_c`, or at which iterations either was resumed.

**Treat `v3_b` and `v3_c` as two seeds of one unknown configuration.** That is all the
evidence supports.

---

## 4. Code defects found by reading the training path

Ranked by expected impact. §4.1–4.3 are candidate explanations for the collapses; §4.4–4.7 are
transformer-specific handicaps; §4.8–4.10 affect *every* run ever done, including the paper's.

---

### 4.1 🔴 A NaN loss randomizes every `nn.Linear` mid-run, silently

**What the code does** — [DMCAgent.py:332-337](agents/DMCAgent.py#L332-L337) (and identically
at :94-99 and :61-66):

```python
loss.backward()
if torch.isnan(loss):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
    self._model.apply(init_weights)
    print(f"Model {self} encountered nan, reset weights to random.")
else:
    nn.utils.clip_grad_norm_(self._model.parameters(), 80)
    self.optimizer.step()
```

**Why it's wrong.** A single NaN batch re-initialises the network and training carries on as
if nothing happened. In detail:

- It resets **only** `nn.Linear.weight`. Not biases. Not `nn.Embedding` (our `card_embedding`),
  not `cls_token`, not any LayerNorm, and — importantly — **not `MultiheadAttention.in_proj_weight`**,
  which is a bare `Parameter`, not a `Linear` module. It *does* hit `out_proj`. The result is
  not a clean restart; it is a chimera of random and trained weights.
- The **RMSprop state survives**. Stale squared-gradient averages, accumulated against the old
  weights, are immediately applied to fresh random ones — a second destabilisation on top of
  the first.
- It sits **inside the per-chunk loop**, so it can fire many times within one call.
- The only evidence is a `print` to a cloud box's stdout, which we no longer have.

**What it cost us.** This matches the `transformer_v3_c` signature almost exactly: a sharp,
permanent, unrecovered drop appearing **simultaneously across all three opponents** at one
checkpoint, followed by a flat lower plateau (the model relearning from a damaged state).
Nothing else in the pipeline produces a discontinuity of that shape. It also fits `v3`'s
slower decay if it fired several times over 340K games.

**Fix.** Never mutate weights on NaN. Skip the batch, count it, and abort loudly if the counter
crosses a threshold — then restart from the last snapshot:

```python
if not torch.isfinite(loss):
    self.nan_batches += 1
    self.optimizer.zero_grad(set_to_none=True)
    continue
```

---

### 4.2 🔴 Optimizer-state load failure is a warning, not an error

**What the code does** — [TrainLoop.py:212-214](TrainLoop.py#L212-L214):

```python
except Exception as e:
    print(f"WARNING: Failed to load optimizer states: {e}")
    print("Model weights are loaded but optimizer state is reset — this may cause instability")
```

**Why it's wrong.** Training proceeds with a **freshly-initialised RMSprop** applied to a
well-trained network. RMSprop normalises updates by `√square_avg`, which starts at **zero**, so
the first few steps are effectively `lr · grad / √(0 + 1e-5)` — enormous. Large steps on a
converged network produce huge losses, which produce infinities, which trigger §4.1, which
destroys the model.

That is a complete, self-consistent causal chain from "resumed a run" to "permanent performance
collapse," using only code that is still in the repo today.

**Context.** Commit `3b3ccb6` ("Fix optimizer state not loading on resume") fixed the
*serialisation* half of this in April. This `except` clause is the half that's still live: it
silently swallows *any* failure, including the dtype/device errors that a mid-experiment
architecture edit would cause.

**Fix.** `raise`, unless an explicit `--allow-fresh-optimizer` flag is passed. If a fresh
optimizer is genuinely wanted, add LR warmup for the first few hundred steps.

---

### 4.3 🔴 Exploration is simultaneously absent *and* stuck at ~40%

Two independent bugs that combine into the worst of both worlds.

#### (a) ε-greedy is a no-op in the main stage

[DMCAgent.py:265-276](agents/DMCAgent.py#L265-L276):

```python
optimal_index = np.argmax(rewards)
if epsilon and random.random() < epsilon:
    if not training and logging.getLogger().level == logging.DEBUG:
        ...
        return sorted_actions[0], action_distribution[optimal_index], entropy
    else:
        return obs.actions[optimal_index], action_distribution[optimal_index], entropy   # greedy!
else:
    return obs.actions[optimal_index], action_distribution[optimal_index], entropy       # also greedy
```

**Both branches return the argmax.** The only difference between them is debug logging. The
random-action branch that exists in the base class ([DMCAgent.py:122-123](agents/DMCAgent.py#L122-L123),
`return random.choice(obs.actions)`) was never carried into `MainModule` when it was overridden.

`git log -L 259,280:agents/DMCAgent.py` confirms this predates all 2026 work — it has been
there since the 2022 code. **The trick-taking stage has never explored, in any run, ever.**
Self-play is fully greedy and therefore self-confirming: if the policy decides early that a
line of play is bad, it will never sample it again to find out otherwise.

#### (b) ε never anneals for the other three stages, and resets on every restart

[TrainLoop.py:218](TrainLoop.py#L218) passes `decay_factor ** (1 / games)` into each actor, and
[TrainLoop.py:67](TrainLoop.py#L67) applies it **once per 10 games, per actor**:

```python
train_sim.epsilon = max(epsilon, train_sim.epsilon / decay_factor)
```

With `--decay-factor 1.2` and `--games 2000`, the per-application factor is
`1.2^(1/2000) ≈ 1.0000912`. Working out how long it takes to anneal ε from `epsilon_start=0.5`
down to the `--epsilon 0.015` floor:

```
ln(0.5 / 0.015) / (ln(1.2) / 2000 / 10 games)  ≈  384,700 games PER ACTOR
                                               ≈  6.9M games total across 18 actors
```

Our runs gave each actor 17K–39K games. So:

| Run | Total games | Games/actor | **ε at end of run** |
|---|---|---|---|
| `transformer_v2` | 300K | 16.7K | **0.43** |
| `transformer_v3` | 700K | 38.9K | **0.35** |

> **ε sat between 0.35 and 0.43 for the entire duration of every experiment we have ever run.**

And because the sampler constructs `Simulation(epsilon=epsilon_start)` unconditionally
([TrainLoop.py:39](TrainLoop.py#L39)) with no reference to `iterations`, **every resume resets
ε back to 0.50**.

**Net effect.** Roughly 40% of all declaration decisions, all 8-per-game kitty discards, and all
chaodi decisions were **uniformly random, in every run, permanently** — while the main stage
explored not at all. That is precisely backwards. A corrupted kitty policy also degrades main-stage
learning indirectly, by handing it worse hands to play from.

---

### 4.4 🟠 The transformer throws away who led each historical trick

**What the code does** — [TransformerModels.py:246](networks/TransformerModels.py#L246):

```python
cards = history[:, :, 4:].reshape(B * 15, 4, 108)   # dims 0:4 are silently dropped
```

Each history row is `[leader_pos(4), self(108), right(108), opposite(108), left(108)]`
([Observation.py:165](env/Observation.py#L165)). The slice `[..., 4:]` discards the
leader one-hot. Nothing else in the file reads `history[:, :, :4]`.

**Why it's wrong.** The LSTM receives all 436 dims and therefore knows who led each past trick.
The transformer does not. And because its four per-trick tokens are ordered by **fixed relative
seat** (SELF/RIGHT/OPPOSITE/LEFT), not by play order, dropping the leader means the model
**cannot reconstruct the order of play, and therefore cannot determine who won any historical
trick.**

In a trick-taking game, "who won the last six tricks" is close to the most important derived
feature there is — it drives inference about who holds what, who is void in which suit, and who
is currently winning. We handed the LSTM that signal and withheld it from the transformer, then
concluded attention was worse.

**Fix.** Concatenate the leader one-hot (and ideally a "this seat won the trick" flag and the
trick's point value) into each history token before projection.

---

### 4.5 🟠 Pre-LN stack with no final LayerNorm

**What the code does** — [TransformerModels.py:192-200](networks/TransformerModels.py#L192-L200):

```python
enc_layer = nn.TransformerEncoderLayer(..., norm_first=True)
self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
#                                        ^ no norm= argument
```

**Why it's wrong.** With `norm_first=True` (Pre-LN), each sublayer normalises its *input*, so
the residual stream itself is never normalised on the way out. PyTorch applies a final
normalisation only if you pass `norm=`. We don't, and the head reads the un-normalised residual
stream directly at `out[:, 0, :]`.

This is the standard Pre-LN footgun: activation magnitude grows with depth. It is fairly benign
at **2 layers** (Experiment 1) and materially worse at **4** (Experiment 3) — which is
consistent with the odd observation that the *bigger* transformer was the *less stable* one, and
it feeds directly into the NaN pathway in §4.1.

**Fix.** `nn.TransformerEncoder(enc_layer, num_layers=L, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)`.

---

### 4.6 🟠 Sixty history tokens, no padding mask

**What the code does** — the padding mask covers only the hand tokens
([TransformerModels.py:310-313](networks/TransformerModels.py#L310-L313)):

```python
key_padding_mask = torch.zeros(B, total, dtype=torch.bool, device=device)
key_padding_mask[:, 1:55] = hand_mask   # hand only; history tokens at 62:122 are never masked
```

**Why it's wrong.** History is **front-padded with zero rows** for tricks that haven't happened
yet ([Observation.py:188-191](env/Observation.py#L188-L191)). A zero cardset does not encode to
zero — it encodes to `cardset_proj(0)` = **the bias vector**, a constant, non-zero, fully-attended
token.

On trick 1 of a game, **56 of the 60 history tokens are meaningless constants** competing for
attention mass with the 4 real ones. The positional encodings at least make them distinguishable,
so the model *can* learn to ignore them — but we're paying capacity and attention budget to learn
something a one-line mask would give for free.

**Fix.** Build a real `key_padding_mask` for history slots from the number of tricks actually played.

---

### 4.7 🟠 Every cardset token is a *linear* function of card counts

**What the code does** — [TransformerModels.py:209-219](networks/TransformerModels.py#L209-L219):

```python
def _enc_cardset(self, x):
    count = x[:, :54] + x[:, 54:]            # (B, 54), values in {0,1,2}
    emb = count @ self.card_embedding.weight # linear map
    return self.cardset_proj(emb)            # another linear map — no activation anywhere
```

**Why it's wrong.** Two linear maps compose to one linear map. So the token for *unplayed cards*,
*kitty*, each *current-trick slot*, the *candidate action*, and all *60 history slots* is a purely
linear function of a card-count vector. No nonlinearity is applied to any card set before it
enters attention.

The LSTM, by contrast, pushes the raw concatenated 873-dim state straight into
`Linear(1129→768) → ReLU → …`. It gets a nonlinear feature extractor over the raw card
indicators immediately; the transformer defers 100% of that work to the attention stack.

For card games this matters more than usual: useful features like "do I hold a tractor in trumps"
or "is this action a pair" are **not linear** in card counts.

**Fix.** Make token encoders a small MLP (`Linear → GELU → Linear`), and give the action token
explicit move-type features (is-pair / is-tractor / length / is-trump / `dominates_all`) rather
than making the model rediscover them from a bag of counts.

---

### 4.8 🟡 Plain DMC has been silently training with a max-entropy bonus at α = 1.0

**What the code does.** [DMCAgent.py:206](agents/DMCAgent.py#L206) initialises
`self.log_alpha = torch.tensor(0.0)`, so **α = exp(0) = 1.0**. Then
[DMCAgent.py:329](agents/DMCAgent.py#L329) applies it **unconditionally**:

```python
loss = self.loss_fn(pred, rewards[idx_chunk] + torch.exp(self.log_alpha) * next_action_entropy[idx_chunk].unsqueeze(1))
```

The `sac` flag gates only the *α update* ([DMCAgent.py:343](agents/DMCAgent.py#L343)), never the
bonus itself. And `Simulation` populates the entropy field regardless of SAC
([Simulation.py:248-252](Simulation.py#L248-L252)).

**Why it's wrong.** Every `--agent-type dmc` run — our transformers, the LSTM baseline, and (by
commit dating, since `ebb64df` predates the published checkpoint) **the paper's 1.2M-game
result** — has been regressing onto `G_t + 1.0 × H(next action distribution)` rather than onto
`G_t`. Nobody chose this.

It's compounded by a second bug at [DMCAgent.py:263](agents/DMCAgent.py#L263):

```python
entropy = -np.mean(action_distribution * np.log2(action_distribution))   # should be np.sum
```

`mean` instead of `sum` divides by N, making the quantity `H/N` rather than `H`. Since that
*decreases* as the number of legal actions grows, the target carries a systematic bias
**toward moves that lead into low-branching states** — a preference for positions where you
have few options, which is close to the opposite of good card play.

**Magnitude.** For a near-uniform distribution over N legal actions the bonus is `log₂(N)/N`:

| N legal actions | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| bonus added to target | 0.46 | 0.33 | 0.22 | 0.13 |

Against returns of ±1–5 (§1.2), that is **roughly 5–30% of the target signal**, and it is
structured noise, not zero-mean noise.

**Fix.** Gate the bonus on `self.sac`, fix `mean`→`sum`, and run the ablation in §6 E5 — the
result is publishable either way.

---

### 4.9 🟡 The observation reports opponent points twice and never shows defender points

**This one is new — I found it while writing this expansion, and it affects every run ever done.**

**What the code does** — [Game.py:145-146](env/Game.py#L145-L146):

```python
defender_points = self.opponent_points,   # ← bug: should be self.defender_points
opponent_points = self.opponent_points,
```

`Game.defender_points` is a real, correctly-maintained quantity — initialised at
[Game.py:50](env/Game.py#L50) and incremented at [Game.py:361](env/Game.py#L361) whenever the
defending team wins a trick containing points. It is simply never passed to the observation.

**Why it's wrong.** `Observation.points_tensor` ([Observation.py:44-55](env/Observation.py#L44-L55))
builds an 80-dim feature whose docstring promises *"First 40 values is for the player's team,
last 40 for the other team"*:

```python
defenders_points_tensor[:(self.defender_points // 5)] = 1
opponents_points_tensor[:(self.opponent_points // 5)] = 1
if self.dealer == RelativePosition.SELF or self.dealer == RelativePosition.OPPOSITE:
    return torch.cat([defenders_points_tensor, opponents_points_tensor])
else:
    return torch.cat([opponents_points_tensor, defenders_points_tensor])
```

Because both inputs are the same number, **both halves are identical**. Three consequences:

1. **40 of the 873 input dims are an exact duplicate** of another 40 — wasted input width in
   every model we've trained.
2. **The team-relative swap at the bottom is dead code.** The model cannot distinguish "my
   team's points" from "their points" from this tensor, which is the entire purpose of the feature.
3. **The defenders' captured-point total is invisible to the network.** With 200 points in the
   deck, `defender_points + opponent_points` tells you how many points are still live in the
   remaining cards — a genuinely important strategic signal that no model has ever received.

**Fix.** One word: `defender_points = self.defender_points`. Then re-verify the swap logic.
Note this changes the input distribution, so it needs a fresh run rather than a resume.

---

### 4.10 🟡 Smaller items, still worth fixing

- **`clip_grad_norm_(..., 80)`** ([DMCAgent.py:339](agents/DMCAgent.py#L339)) — for an MSE loss
  on ±5 targets, typical gradient norms are orders of magnitude below 80, so the clip is
  effectively **inactive**. It provides no protection against the spikes in §4.1/§4.2.
  Tighten to ~1.0.
- **`strict=False` on every `load_state_dict`** ([DMCAgent.py:388](agents/DMCAgent.py#L388),
  :396, :404, :423) — silently tolerates missing keys, so an architecture edit loads a
  partially-random model with no complaint. Use `strict=True` and handle migrations explicitly.
- **LR is hardcoded** at `RMSprop(lr=1e-4, alpha=0.99, eps=1e-5)`
  ([DMCAgent.py:40](agents/DMCAgent.py#L40)) with no CLI flag, no schedule and no warmup — and
  was never re-tuned when the architecture changed.
- **`tau=0.1` soft update runs once per queue item**, not per gradient step
  ([DMCAgent.py:349-350](agents/DMCAgent.py#L349-L350)), so the actors' behaviour policy lags
  the learner by an amount that depends on queue throughput — i.e. **the effective algorithm
  changes when you change the number of actors**. Worth pinning down before comparing runs made
  on different machines.

---

## 5. The throughput bug that unlocks everything else

Training is actor-bound (§1.3: queue wait is 86–92% of wall clock), so sampling speed *is* the
experiment budget. There is a large and easy win sitting in the hot path.

**What the code does** — [DMCAgent.py:260](agents/DMCAgent.py#L260):

```python
x_batch, history_batch, *_ = self.prepare_batch_inputs([(obs, a, 0, None) for a in obs.actions])
```

For a decision with N legal actions this builds an N-row batch from **one** observation. Inside
`prepare_batch_inputs` ([DMCAgent.py:219-257](agents/DMCAgent.py#L219-L257)), every row
recomputes the full state. But across those N rows, **only the last 108 dims (the action) and
one dim (`dominates_all`) actually differ.** Everything else is identical — and `history_batch`
is **N identical copies of the same 15×436 tensor**.

Two separate costs:

1. **`obs.historical_moves_dynamic_tensor` is recomputed N times per decision.** It's a plain
   `@property` ([Observation.py:177-192](env/Observation.py#L177-L192)) that runs a Python loop
   building up to 60 card-set tensors via `get_dynamic_tensor`. We call it once per *candidate
   action*, for the same `obs` object, inside the process that is the bottleneck. With N≈10–50
   actions × ~25 decisions × 4 seats per game, that is **thousands of redundant tensor
   constructions per game**.
2. **The transformer re-encodes 121 identical state tokens N times**, including all 60 history
   tokens. This is why transformer inference is 0.36 s/game versus the LSTM's 0.10 s — it's not
   that attention is inherently slow here, it's that we run the state encoder 10–50× more often
   than necessary.

**Two fixes, both mechanical:**

- **Cache the history tensor** (`functools.cached_property`, or compute once in `act()` and
  `expand` it across the batch). Costs nothing, speeds up *both* architectures, and directly
  buys experiment throughput.
- **Restructure as encode-state-once, score-actions-cheaply.** Run the encoder a single time
  per decision, then score all N candidate actions against the cached state representation via
  cross-attention or a small action head — turning per-decision cost from `O(N × encoder)` into
  `O(encoder + N × head)`.

That second change is the one that makes everything else affordable: it could plausibly make the
transformer *faster per decision than the LSTM*, which buys the compute headroom to afford the
much larger Q-head that §2 says we need.

---

## 6. How to actually beat the LSTM

Ordered by expected value per GPU-hour.

### E0 — Instrumentation and hygiene *(do first; ~half a day, no GPU)*

Non-negotiable prerequisites. Without these, no future run is interpretable.

1. **Never randomize weights on NaN** (§4.1). Skip the batch, count it, abort loudly past a
   threshold, restore from snapshot.
2. **Fail loudly on optimizer-load failure** (§4.2) — `raise` unless `--allow-fresh-optimizer`.
3. **Persist diagnostics to `stats.pkl`:** mean/max loss, gradient-norm percentiles,
   `nan_batches`, current ε, queue depth. Plot them beside the eval curves.
4. **Tighten `clip_grad_norm_` to ~1.0**; add `--lr` and `--optimizer` flags.
5. **Fix the eval protocol:** make `--eval-size` **per opponent** rather than divided; use ≥500
   games/opponent for headline numbers; drop or rename the meaningless cross-opponent aggregate;
   log per-opponent Wilson confidence intervals.
6. **Sync `command.txt`, `state.pkl` and `snapshots/` back** from the cloud box along with stats.

### E1 — The fair fight *(the experiment we thought we ran)*

Fix the transformer-specific handicaps, then re-run the head-to-head with **one variable**:

- Add `norm=nn.LayerNorm(d_model)` (§4.5).
- Restore the leader one-hot, plus a trick-winner flag, in history tokens (§4.4).
- Add a real history padding mask (§4.6).
- Give token encoders a nonlinearity, and the action token explicit move-type features (§4.7).
- **Set `dropout=0.0`** to match the LSTM, and re-tune the optimizer for the transformer
  (AdamW, lr≈3e-4, 2K-step warmup, cosine decay, wd 0.01).
- Apply the E2 head rebalance below.

Run **2 seeds per arm** at equal games with an identical eval protocol. Report level rate vs
`strategic` and vs a **frozen** LSTM checkpoint, with confidence intervals.

### E2 — Move the capacity into the Q-head ⭐ *highest-value single change*

This follows directly from §2. Replace the 8.3K-param head with one in the LSTM's weight class
(`d_model→512 → ReLU → 3–4 × (512→512, ReLU) → 512→1`, ≈1.1M params), **and give the head direct
access to the raw state** so it isn't reasoning through a 7× bottleneck:

```python
head_in = torch.cat([out[:, 0, :], x], dim=-1)   # 128-dim CLS + the raw 873-dim state
return self.head(head_in)
```

This mirrors exactly what the LSTM does (`hstack([x, history_out])`) while keeping attention over
history. **Prediction: worth more than every other change in this document combined.**

### E3 — The hybrid *(cleanest test of the original thesis claim)*

Take [`networks/Models.py`](networks/Models.py) **unchanged** and replace *only*
`nn.LSTM(436, 256)` with a small transformer encoder over the 15 history steps, pooled to 256
dims. Everything downstream — the whole 1129→768→512→512×3→1 head — stays byte-identical.

This isolates one question: *is attention better than recurrence for encoding this game's
history?* That is the actual thesis-relevant claim, and **it has never been tested.** It's also
cheap: one class, one file, same hyperparameters. If E2 and E3 both win, you have a clean story:
attention helps, but only once the head has the capacity to use it.

### E4 — Restore exploration *(likely a step change for all architectures)*

1. Fix the main-stage ε no-op (§4.3a) — restore the `random.choice(obs.actions)` branch.
2. Anneal ε against **global** games, not per-actor batches; persist ε across resumes.
3. Sweep main-stage ε ∈ {0.01, 0.03, 0.10}, and **lower the other three stages** — ~40% random
   kitty discards for entire runs is almost certainly destroying the kitty policy.
4. Consider Boltzmann sampling over Q-values (already implemented in the 2022 codebase, unused
   since) now that we have eval opponents sensitive enough to detect the difference.

### E5 — Ablate the accidental entropy bonus (§4.8)

Three arms at equal games: α=1.0 with `mean` (status quo), **α=0 (pure DMC)**, α=0.1 with a
corrected `sum`. Cheap, and it settles whether a bug has been silently shaping every result in
this project's history — including the published 97.7%. Interesting either way.

### E6 — Fix the points observation (§4.9) and re-baseline

One-word fix, but it changes the input distribution, so it needs a fresh run. Worth folding into
whichever arm of E1 wins rather than running standalone.

### E7 — Then, and only then, scale

With E0–E2 landed and the §5 throughput fix in, scale d_model/layers/head together and train to
paper scale (700K–1.2M games), snapshotting every 50K, with CI-annotated curves.

---

## 7. Suggested order of work

| # | Content | Cost | Why in this position |
|---|---|---|---|
| 1 | **E0** instrumentation + eval protocol | ~0.5 day, no GPU | Nothing below is interpretable without it |
| 2 | §5 history-tensor caching | ~1 hour | Speeds up every subsequent run |
| 3 | **E2** big Q-head + **E1** transformer fixes | ~1 day | The main hypothesis |
| 4 | **E1** A/B, 2 seeds, 300K games | 2 × ~11 h, ≈$8 | First trustworthy number in the project |
| 5 | **E4** exploration fixes | ~0.5 day + 1 run | Orthogonal to E1–E2, probably large |
| 6 | **E3** hybrid + **E5** entropy ablation | 2 runs | Thesis-grade controlled claims |
| 7 | **E6/E7** points fix, scale to 1.2M games | ≈$15 | Headline result |

Total GPU spend to a trustworthy answer is on the order of **$30–50** at Vast.ai rates. The
binding constraint here is engineering time, not compute — which is itself worth remembering:
Experiments 1–3 spent most of their effort on infrastructure and none on controls.

---

## 8. Confidence levels

Being explicit so future-you knows what is established and what is speculation:

- **Verified by reading the code** *(high confidence)*: every defect in §4, the throughput
  redundancy in §5, and the §2 parameter accounting — which reproduces the documented
  124K / 860K / 2.76M totals exactly, so the arithmetic is sound.
- **Strong inference** *(medium-high)*: §4.1 and/or §4.2 caused the `v3` decay and the `v3_c`
  step-change. The signature fits — sharp, permanent, simultaneous across all three opponents —
  and no competing explanation has been found. But the logs that would prove it were never
  retained, so this stays a hypothesis.
- **Prediction, untested** *(medium)*: that E2 (Q-head capacity) is the dominant lever. It
  follows from the parameter table and from the fact that the Exp-3 encoder already exceeds the
  LSTM's, but it is a hypothesis until run.
- **Unknown**: what actually differed between `transformer_v3_b` and `transformer_v3_c`.
  Provenance is lost (§3.5); treat them as two seeds of one configuration.

One caveat worth stating plainly: §4.3a, §4.8 and §4.9 affect the **LSTM baseline and the
original paper equally**. They are not explanations for why the transformer underperformed —
they are reasons why *every* number this project has ever produced is lower than it should be.
The transformer-specific handicaps are §4.4–4.7 plus the §2 head imbalance.
