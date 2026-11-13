# Summary: What We've Tried to Improve Model Performance

**Last updated:** 2026-11-12 (see §7 for this round's changes)

This document consolidates everything attempted so far to improve ShengJi++'s playing
strength, drawing on `docs/experiment_1.md`, `docs/experiment_2.md`, git history, and
the raw stats in `exps/*/stats.pkl` (experiment 3 was never written up as a doc, so its
results are reconstructed here from the pickled training logs).

---

## 1. Inherited from the original ShengJi+ project (2022–2023)

Before this thesis picked the project back up, the original authors (`TheMoon2000`,
`Leotrinos`) explored the RL algorithm/exploration space and landed on the setup this
codebase still uses:

- **Algorithms tried:** tabular/DQN-style Q-learning → **Deep Monte Carlo (DMC)** →
  Soft Actor-Critic (SAC) → maximum-entropy DMC. DMC is what stuck and is still the
  default (`agents/DMCAgent.py`).
- **Exploration strategies tried:** Boltzmann exploration, hash/count-based novelty
  exploration (`env/StateAutoEncoder.py` was built to support this by compressing
  states into a hashable representation).
- **Reward/feature shaping:** kitty points added to the point arrays, reward
  recalculation, fixed-deck and level-count features, a "combo penalty" option to
  discourage over-using combo moves.
- **Result:** the original paper's checkpoint reached **97.7% level rate vs. the
  random baseline after ~1.2M self-play games**, using DMC with combos enabled. This
  is the number the current experiments are ultimately trying to reproduce/exceed
  under a new architecture.

None of this predates a rewrite — it's still the foundation. The current work has not
revisited SAC, max-entropy DMC, or count-based exploration; all current experiments
are plain DMC + ε-greedy.

---

## 2. Current thesis work (2026) — infrastructure and Experiment 1: LSTM → Transformer

Full detail in `docs/experiment_1.md`. Before any training run, a batch of engineering
bugs had to be fixed (stray `turtle` import breaking headless Linux, unpicklable
dynamically-loaded models breaking multiprocessing, `pickle`-based model cloning
replaced with `copy.deepcopy`).

**Architecture change:** replaced the LSTM-based `MainModel` with a Transformer
encoder (`networks/TransformerModels.py`) that attends over the previous 15 rounds of
move history instead of compressing them into a hidden state.

**CUDA OOM fixes**, in order of impact:
- Chunked legal-action evaluation (batches of 32) instead of one huge batched forward pass.
- Model size cut ~7× (860K → 124K params) to fit many concurrent actor processes in 24GB VRAM.
- `torch.cuda.empty_cache()` before spawning eval processes (allocator fragmentation).
- Unset `expandable_segments` (incompatible with `torch.multiprocessing.spawn`).

**Training throughput work:**
- Moved from RunPod to Vast.ai after discovering RunPod hard-caps containers at
  ~27.2 logical cores regardless of advertised vCPU count.
- `OMP_NUM_THREADS=1` was the single biggest unlock — without it, 14+ actor processes
  exhaust the OS thread budget via internal OpenMP thread spawning.
- Pre-tensorization: moved `prepare_batch_inputs` out of the single main process and
  into the parallel actor processes. This cut the main process's "learn time" share
  from 63–68% down to 1–7% and made GPU utilization no longer the bottleneck.
- Final config: 18 actor / 6 eval processes on a Vast.ai EPYC 9554 + RTX 4090 box,
  sustaining **7.4 games/s** at ~$3.60 for a 300K-game, 11-hour run.

**Result (124K-param transformer, 300K games):** 88% win rate / 92.2% level rate vs.
random; 56.9% win rate vs. the rule-based `StrategicAgent`. Win rate vs. random
plateaued after ~20K games — an early sign the metric itself was saturating, not that
the model stopped improving.

---

## 3. Experiment 2: LSTM baseline reproduction

Full detail in `docs/experiment_2.md`. Reran the *original* LSTM architecture
(2.76M params in `MainModel`, vs. 124K in the Experiment 1 transformer — a 22× gap)
under the same hyperparameters, to get an apples-to-apples baseline.

**Finding:** even cut short at 196K games (vs. the paper's 700K), the LSTM reached
94.3% level rate — already ahead of the transformer's 92.2% at 300K games (50% more
training). **Conclusion: at these sizes, model capacity dominated the architecture
choice** — the transformer's theoretical edge (attention vs. fixed hidden state over
history) never got to show up because it was ~22× smaller.

This directly motivated scaling the transformer back up.

---

## 4. Experiment 3: Scaled-up transformer + multi-opponent eval (undocumented, reconstructed from stats.pkl)

No `experiment_3.md` was written for this round; the following is pieced together from
`exps/transformer_v3*/stats.pkl` and the relevant commits.

**Changes made (per commit history, 2026-04-01 → 04-05):**
- Scaled the transformer back up to d_model=128, nhead=8, num_layers=4 (~860K params,
  matching the "Experiment 2 planned" config), now that chunked inference +
  pre-tensorization removed the original OOM pressure.
- **Fixed a real bug:** optimizer state was not being restored on resume (switched to
  `torch.save`/`torch.load` for optimizer state). This shipped *after* the `v3` run
  had already scaled up, and before `v3_b`/`v3_c`.
- Added versioned weight snapshots every 100K games (`<model_folder>/snapshots/<n>/`)
  so a good checkpoint can't be silently overwritten by a later, worse one.
- Added `--eval-agents` for evaluating against multiple opponents per checkpoint in
  one run (`random`, `strategic`, `model:<folder>`), directly addressing Experiment 1's
  "next steps" note that random-only eval is uninformative once the model beats it
  consistently.

**`transformer_v3` (860K params, 700K games, random-only eval):**

| Games | Level rate vs. random |
|---|---|
| 2K | 82.2% |
| ~240K–360K (peak) | 92–95.5% |
| 700K (final) | 82.6% |

The run peaked around 240–360K games and then **degraded steadily for the remaining
~340K games**, ending barely above where it started. This is the run that shipped
*before* the optimizer-resume-state bugfix — the timing strongly suggests the
mid-training decline is an artifact of the optimizer state being dropped across
process/resume boundaries, though this was never confirmed with a controlled rerun.

**`transformer_v3_b` (318K games, multi-opponent eval, post-bugfix):** improved
steadily throughout — win rate vs. `lstm_baseline` rose from 31/100 to 46/100, vs.
`strategic` from 26/100 to 43/100, vs. `random` level rate reaching ~90%+.

**`transformer_v3_c` (482K games, latest, multi-opponent eval):** tracks `v3_b`
closely and slightly ahead for the first ~320K games (e.g. ~90% level rate vs. random,
~55–60% vs. `strategic`, ~40–50% vs. `lstm_baseline`). **Around the 320K–330K game
mark, performance drops sharply and does not recover**: level rate vs. `strategic`
falls from ~90% to ~85% and stays there, vs. `lstm_baseline` falls from ~45% to
~30–35%, and average opposition points drop by ~10 points in the same window. The
cause is not documented/isolated — no command log or config diff was kept for `v3_b`
vs. `v3_c`, so it's unclear whether this is a resume artifact (same failure mode as
`v3`?), a hyperparameter change, or an eval-opponent change partway through. **This is
the most concrete open lead for the next round of work.**

---

## 5. Standing takeaways

1. **Model capacity mattered more than architecture** at the sizes tried so far — the
   860K-param transformer was built specifically to close the gap with the 2.76M-param
   LSTM baseline; results above are its first real test.
2. **Evaluating against random is not enough** once a model is good — it saturates
   early (~20K games) and stops reflecting further improvement. `strategic` and
   `model:<checkpoint>` opponents (added in Experiment 3) are the useful signals now.
3. **Optimizer state must survive resumes.** This was silently broken for some portion
   of the project's training history and is suspected (not proven) to explain the
   `transformer_v3` degradation after ~360K games.
4. **Infra/throughput fixes compound:** `OMP_NUM_THREADS=1` + pre-tensorization +
   Vast.ai over RunPod took throughput from ~2.3 games/s to ~7–9 games/s, which is what
   makes 300–700K-game runs affordable (~$3–5 each) rather than prohibitive.
5. **No experiment has yet reached the original paper's scale** (1.2M games) or beaten
   its 97.7% level-rate-vs-random result — the closest so far is the LSTM baseline at
   94.3% after only 196K games (stopped early, not saturated).

---

## 6. Open threads / natural next steps

- **Root-cause the `transformer_v3_c` regression at ~320K games** — rerun with
  snapshots + a saved command log so the exact config and resume points are
  reproducible, and check whether reverting to the pre-drop snapshot and continuing
  reproduces or avoids the drop.
- **Finish a full run to paper-scale (~700K–1.2M games)** for the 860K-param
  transformer without a mid-run collapse, to get a real head-to-head against the
  2.76M-param LSTM and the original paper's number.
- **Revisit exploration strategy** (Boltzmann / hash-count, both implemented in the
  original 2022 codebase but unused since) now that eval infrastructure can actually
  detect whether it helps past the random-baseline plateau.
- **Curriculum learning against `strategic`/checkpointed opponents**, as flagged in
  Experiment 1 — self-play alone may be leaving headroom on the table given the
  56.9%–~60% win rates against the rule-based baseline so far.

---

## 7. This round (2026-11-12): fixing the bugs that made Experiments 1–3 uninterpretable

A full read-through of the training path (`docs/improvement_plan.md`) turned up several bugs
that affected every run above, not just the transformer: the main stage never explored
(ε-greedy was a no-op), a NaN batch silently re-randomized weights instead of being skipped, a
resumed optimizer falling back to fresh state was a warning instead of a hard failure, an
accidental entropy bonus was added to every plain-DMC target, and the points-tensor observation
fed the model the same number for both teams. All five are now fixed — see the status table at
the top of `improvement_plan.md` for the complete list and exact commit references.

On the architecture side specifically, the Experiment 1→3 "LSTM vs Transformer" comparison
changed six things simultaneously (§3.1): history encoder, Q-head capacity (2.05M vs 8.3K
params!), head input bottleneck, per-cardset nonlinearity, dropout, and optimizer tuning — then
attributed the whole gap to architecture. The Q-head capacity and head-bottleneck fixes alone
(§2/E2) are the predicted highest-value change in the entire writeup. All five transformer-only
confounds are now fixed, training diagnostics (loss/grad-norm percentiles, NaN counts, ε) are
persisted to `stats.pkl` instead of being discarded after printing, and the eval protocol no
longer silently divides `--eval-size` across opponents.

**None of this was executed** — torch can't load on the machine this work was done on (an OS
policy blocks the DLL), so every change was verified by static read-through, and the
torch-independent pure-Python logic (`epsilon_at`, `wilson_interval`) was extracted and run
standalone instead. **`docs/experiment_4_runbook.md` starts with a mandatory smoke test** for
exactly this reason — run it before committing GPU money to a full experiment.

This also surfaced a gap worth flagging: `exps/lstm_baseline/` only has `stats.pkl` and a PNG
locally, no weight checkpoint or command log (the exact §3.5 provenance-loss problem, now
happening to the LSTM baseline too). The runbook calls this out explicitly rather than silently
dropping the LSTM from the next comparison.
