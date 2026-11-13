# Experiment 4 Runbook: the fair fight

**Date:** 2026-11-12
**Companion to:** [`improvement_plan.md`](improvement_plan.md) (full diagnosis), [`summary.md`](summary.md)
**Goal:** run the controlled transformer-vs-LSTM comparison that Experiments 1–3 never actually
ran, now that the E0 hygiene pass and most of E1's transformer-specific fixes have landed (see
the status table at the top of `improvement_plan.md`).

This is deliberately short — it's a checklist, not an analysis. Read `improvement_plan.md` first
if you haven't already.

---

## 0. Before spending any GPU money: a local smoke test

None of this round's changes were executed — torch can't load on this machine (Windows
Application Control policy blocks `c10.dll`), so everything was verified by static
read-through and, where possible, extracting the pure-Python logic (`epsilon_at`,
`wilson_interval`) and running it standalone. The architecture changes in
`networks/TransformerModels.py` in particular (new head shape, history padding mask, leader
one-hot) were never run through an actual forward pass.

**Do not skip this step.** On whatever machine actually has a working torch install:

```bash
python -m unittest discover -s tests -v
python TrainLoop.py --model-architecture transformer --model-folder /tmp/smoke_test \
    --games 20 --single-process --actor-processes 1 --eval-size 2 --max-games 20 --verbose
```

This should complete in well under a minute and exercises every changed code path (model
construction, the new head/mask, a few training steps, eval). If it throws a shape error, that's
the new head/history-mask code — check `_TransformerBase._forward_impl` in
`networks/TransformerModels.py` against the comments there before touching anything else.
Delete `/tmp/smoke_test` afterwards; it's not a real run.

## 1. Known gap: no usable frozen LSTM checkpoint locally

`exps/lstm_baseline/` only has `stats.pkl` and a PNG — no `.pt` weights, no `command.txt`, no
`state.pkl` (this is exactly the §3.5 provenance loss the improvement plan flags). If those
weight files still exist on whatever box trained it, sync them down into `exps/lstm_baseline/`
(`declare.pt`, `kitty.pt`, `chaodi.pt`, `main.pt`, `Models.py`, `state.pkl`, `command.txt`) before
running E1 below, so `--eval-agents model:exps/lstm_baseline` works. If they're gone, drop that
eval agent and compare against `strategic` + `random` only for this round — still informative,
just missing the LSTM head-to-head.

## 2. The E1 run itself

Two seeds, equal games, fixed eval protocol. Each `vast.ai` box runs one seed.

```bash
python TrainLoop.py --model-folder exps/transformer_v4_seed1 \
    --model-architecture transformer \
    --random-seed 1 \
    --discount 0.95 --games 2000 --max-games 300000 \
    --epsilon 0.02 --epsilon-start 0.5 \
    --optimizer adamw --lr 3e-4 --weight-decay 0.01 --warmup-steps 2000 --dropout 0.0 \
    --grad-clip 1.0 \
    --eval-size 500 --eval-agents random strategic model:exps/lstm_baseline \
    --enable-combos --combo-penalty 0.01 \
    --actor-processes 18 --eval-processes 6
```

Run the same command with `--random-seed 2 --model-folder exps/transformer_v4_seed2` on a
second box (or sequentially on the same one).

Notes on the flags that changed this round:
- `--optimizer adamw --lr 3e-4 --weight-decay 0.01 --warmup-steps 2000` and `--dropout 0.0`
  implement the E1 recipe (§6 E1): AdamW instead of the never-tuned inherited RMSprop LR, and
  dropout matched to the LSTM's 0.0 for a controlled comparison. `--warmup-steps` is in
  *optimizer steps*, not games — 2000 is a reasonable start; watch `grad_norm_stats` in the
  first checkpoint's `stats.pkl` entry and raise it if gradients are still spiky after warmup
  ends.
- `--eval-size 500` is now **per opponent** (changed this round — previously divided across
  `--eval-agents`), so this is 500 games each vs random/strategic/the LSTM, not 500 total.
- `--grad-clip 1.0` and `--allow-fresh-optimizer` (not passed here — a resume failing to load
  optimizer state will now raise rather than silently continue).
- If you need to resume a crashed run, just rerun the same command with the same
  `--model-folder`; `iterations` and ε both pick up correctly from `state.pkl`/`stats.pkl`
  without a `--epsilon-start` reset (the ε-annealing-on-resume bug this round fixed).

## 3. While it's running / when it finishes

- Check `stats.pkl` periodically (`python plot_stats.py --model-folder exps/transformer_v4_seed1`)
  — the per-opponent level rate now comes with each checkpoint's `nan_batches` and
  `grad_norm_stats` alongside it. A `nan_batches` count that's nonzero anywhere is worth
  investigating before trusting the curve past that point; the run will abort on its own if it
  exceeds 100 for any stage module.
- **Sync back more than stats this time** (§3.5's provenance loss, E0 item 6): copy
  `command.txt`, `state.pkl`, `Models.py`, `main.pt`/`declare.pt`/`kitty.pt`/`chaodi.pt`, and the
  `snapshots/` folder off the cloud box along with `stats.pkl` and the PNGs, for both seeds.
  Without this, the next person can't tell what produced the numbers either.
- Compare the two seeds against each other before drawing any conclusion from either one — §3.4
  found run-to-run variance large enough to swamp most single-run effects in this project's
  history.

## 4. What this round deliberately did NOT do

So expectations are calibrated — these are documented, not forgotten:

- **ε is still one shared value across all four stages.** The plan's E4 suggested a sweep with
  a *lower* ε for declare/kitty/chaodi than for the main stage; that would need splitting
  `Simulation.epsilon` into per-stage values, which wasn't done this round. Main-stage ε-greedy
  itself is fixed (§4.3a) and now anneals correctly (§4.3b) — just not stage-differentiated.
- **No trick-winner flag or point-value feature in history tokens** (§4.4's "ideally" extra,
  not the mandatory leader one-hot, which is fixed). Computing it correctly needs
  `CardSet.round_winner`-style logic, which isn't vectorizable Python inside `forward()` without
  either precomputing it in `Observation` (a layout change affecting the LSTM too) or a slower
  per-row loop. Left as a follow-up.
- **LR warmup is not persisted across resumes.** Each process restart replays the first
  `--warmup-steps` optimizer steps at a reduced LR before ramping back to `--lr`. This is a
  minor, self-correcting perturbation on resume, not a destabilizing one like the optimizer-state
  bug this round fixed — but if you resume often, be aware the warmup replays each time.
- **`tau=0.1` soft-update-per-queue-item** (§4.10) is unchanged — the eval/behavior policy's lag
  behind the learner still depends on actor throughput, so comparisons across machines with
  different actor counts aren't perfectly clean.
- **Boltzmann sampling over Q-values** (E4.4) — exists unused in the 2022 codebase, not wired
  into this round's changes.
- **`agents/DQNAgent.py` is untouched.** CLAUDE.md's documented path is DMC; the DQN/SAC agent
  wasn't part of this round's scope.
- **Oracle mode (`--oracle-duration`) + `--model-architecture transformer`** was already an
  unsupported combination before this round (the transformer ignores `use_oracle`) and still is
  — the input width would mismatch the head's fixed `raw_x_dim`. Leave `--oracle-duration 0`
  (the default) with the transformer.

## 5. Cost/time expectations

Per the improvement plan's own estimate (§7): ~$8 for 2 seeds × 300K games at Vast.ai rates,
assuming the ~7-9 games/s throughput the project has previously achieved with
`OMP_NUM_THREADS=1` set and 18 actor processes. Confirm that env var is still set on whatever box
you provision — it isn't code, so there's nothing here to have silently regressed, but it's easy
to forget on a fresh box.
