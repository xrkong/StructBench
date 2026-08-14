# 0052 — Hybrid local+global backbone: the MP-Transolver dual-track (Design A)

**Status**: Proposed — drafted by Claude Code, prototype landed on
`feat/hybrid-local-global`; the human finalises
**Type**: Durable
**Date**: 2026-08-15

## Context

Across the multi-method extensions (ADR-0046/0047/0048) the operator families
(Transolver, GeoFLARE) win **interpolation** decisively — Transolver beats the
blessed CGN 4–6× on Taylor and notch `test_interp` — but lose the notch
**off-grid probe**, where message-passing MGN is strongest (probe ~0.27 mm vs
operators 0.58–0.67; the operator QoIs blow up on the off-grid case). The G3
investigation (design record in `scratch/2026-08-14-g3-*.md`) asked whether a
**local (message-passing) + global (operator)** hybrid can hold the operator's
interpolation win *and* recover the probe.

Two independent passes — a literature scan and an internal-fit analysis
grounded in the actual model code — **converged** on one architecture:
**Design A, the MP-Transolver dual-track**. The decisive read of our own
evidence: GeoFLARE already carries dynamic ball-query geometry *descriptors*
and still loses the probe, so the missing ingredient is **state-carrying local
message passing over relative position** (`MGNet.processor_block`), not more
geometry context.

A P3 adversarial critique (`scratch/2026-08-14-g3-P3-critique.md`) then
attacked the causal premise: if the probe fails purely by **rollout
accumulation** (our own summary: "one-step ~0.0035 near-optimal everywhere;
stability is the open problem"), a local branch is aimed at the wrong failure,
and a stability intervention (the ADR-0050/0051 k-frames axis) would be the
right lever instead. It demanded a **zero-GPU one-step-vs-rollout decomposition
gate** on the existing blessed Transolver/MGN checkpoints before any prototype
train, plus five concrete must-fix wiring changes.

**The gate diagnostic greenlit Design A.** The probe gap is **~2.4× on the
one-step (representation) metric**, not pure accumulation — rollout
accumulation itself is roughly **method-independent** across families. A gap
that already exists at one step is a *representation* deficit an operator's
missing local pathway can plausibly close, so a representation intervention
(Design A) is on-target; a pure stability intervention is not the whole story.

## Decision

Implement **Design A** as a prototype behind an off-by-default config switch,
with all five P3 must-fix changes. The MP branch is a **third residual sublayer
inside `TransolverBlock`**, parallel to Physics-Attention and gated:

```
fx = attn(ln_1(fx), segments)          + fx    # GLOBAL operator (unchanged)
fx = gate * mp(ln_local(fx), graph)    + fx    # NEW LOCAL MP branch
fx = mlp(ln_2(fx))                     + fx    # FFN (unchanged)
```

`mp` = `LocalMessagePassing`: an edge MLP on `[edge_feats, fx[send], fx[recv]]`
→ receiver-side `index_add_` → node MLP on `[fx, agg]`, reusing MGN's
`build_mlp`. The five P3 fixes, each load-bearing:

1. **Middle block only, never the last.** The final `TransolverBlock` carries
   the no-residual decoder head; an MP branch there is a wiring trap. MP goes on
   `hybrid_blocks` MIDDLE blocks (`hybrid_block_indices`, default 1 = block
   `L//2`), never block 0 or the decoder block.
2. **k-capped radius edge-index, written as new code.** `world_edges` is
   uncapped `O(P²)` (the ~760k-step world-edge OOM) and `ball_query` returns
   padded *coordinates*, not an `edge_index` — neither is reusable verbatim. A
   new builder (`graph_ops.build_radius_graph`) emits a top-`max_neighbors`
   within-radius `(2, E)` index (`E ≤ P·k`) plus MGN-style `[x_ij, |x_ij|]`
   edge features, **per-example partitioned** so no edge crosses an example
   boundary (the invariant that keeps `index_add_` segment-safe). Built **once
   per step** from the current positions (`x_t` eval / `x_last` train) and
   reused across every hybrid block, exactly as MGN reuses its world-edge index.
3. **Mean-normalized aggregate.** The receiver aggregate is divided by per-node
   in-degree (`clamp_min 1`), not summed. A raw sum scales messages with
   wildly-varying degree (Taylor's mushroom-foot compaction spikes degree
   10–100×), which would destabilize the operator's interpolation win when the
   gate opens; mean-normalization removes that coupling.
4. **Small-nonzero LayerScale gate, no weight decay.** A per-channel gate is
   initialized to `1e-3` (NOT exactly 0 — an exactly-0 gate gives the MP-branch
   parameters zero gradient at step 0 and, with weight decay, can be pulled back
   to 0 and never engage). The gate and the whole MP branch are **excluded from
   AdamW weight decay** (a second param group; the vanilla single-group recipe
   is untouched when the branch is off).
5. **One optional `graph=` arg, back-compatible.** `TransolverNet.forward` and
   `TransolverBlock.forward` gain a single optional
   `graph=(edge_index, edge_feats)`; `graph=None` with `hybrid_mp=False` is
   **byte-identical vanilla Transolver** (no MP submodules built, no extra
   params, no forward change), so existing checkpoints, tests, and the
   k-frames / φ-conditioning branches all load and run unchanged. A legacy
   `config.json` lacking the hybrid keys reconstructs as `hybrid_mp=False`.

Config: four `TransolverConfig` knobs — `hybrid_mp=false` (default off =
vanilla), `hybrid_radius`, `hybrid_max_neighbors=32`, `hybrid_blocks=1`. The
strict exact-keys loader requires them in every transolver TOML (added to all
16 existing configs + the test fixture) and **rejects `hybrid_mp=true` without a
positive `hybrid_radius`** (the branch has no graph otherwise).

**Scope: prototype, not a blessed method.** This ADR authorises the
architecture, the two P2 sanity fleets, and the iteration ladder — not a
results-registry entry. Blessing is a later ADR gated on the KPI below.

## Alternatives considered

- **Mechanism C — global-augmented MGN** (a `PhysicsAttentionIrregularMesh`
  residual injected into the MGN processor). The mirror arm: it starts from the
  probe *winner* and only needs to not break it while adding Taylor/interp
  capacity, at lower cost-to-KPI and with no graph/API code. Kept as the
  second-priority confirmation arm; Design A is prototyped first because it
  starts from the interpolation winner and patches one localized failure rather
  than rebuilding, and its dynamic edges sidestep MGN's Taylor 31× mesh-stretch
  failure by construction.
- **Mechanism B — local-feature-conditioned slices** (port GeoFLARE's
  ball-query descriptors onto Transolver's slice assignment). Rejected as a
  standalone fix: GeoFLARE already carries exactly these descriptors and still
  loses the probe — it adds geometry *description*, not local *state exchange*.
  Retained only as a cheap ablation/control.
- **A pure stability intervention (k-frames / pushforward, ADR-0050/0051).** Not
  chosen for the probe because the gate diagnostic located ~2.4× of the gap at
  one step (representation), which a stability lever does not touch. Orthogonal
  and composable with this branch.
- **Sum aggregation / exactly-zero gate / MP on the last block / uncapped
  `world_edges`** — each rejected by a specific P3 failure mode (§3–4 above).

## Consequences

- Transolver gains an off-by-default local message-passing capability with
  **zero impact on the vanilla path** (byte-identical, param-count-identical,
  legacy-checkpoint-safe) — verified by a CPU smoke: the vanilla path is
  unchanged; a hybrid forward/backward is finite and a tiny batch overfits; the
  **MP edge MLP receives nonzero gradient after one backward** (the key check —
  it would be exactly 0 with a zero gate); the graph stays degree-bounded on a
  contact-dense cloud; and the MP branch demonstrably changes the one-step
  prediction. An end-to-end `train()` → `evaluate()` hybrid smoke exercises the
  batched-graph training path, the weight-decay-excluded param group, and the
  eval rollout.
- **Validation plan (signed, before compute).** Iteration ladder, each step:
  CPU smoke → **Taylor 2×60k** (`transolver-hybrid-s1/s2`, off the N02+VH
  recipe) → if it matches Transolver's interp *at matched 60k budget* (the
  must-not-regress gate; compare against a matched-budget plain-Transolver
  control, NOT the blessed full-budget number) → **notch 2×60k**
  (`transolver-hybrid`, off the plain-VH recipe) for the probe KPI. Predictions:
  Taylor interp holds (~0.15–0.21 val, operator backbone untouched, dynamic
  edges avoid the 31× stretch); the notch off-grid probe improves toward MGN's
  0.27 without the operator QoIs blowing up. If one middle block moves the
  probe, scale `hybrid_blocks`; if it does not, the hypothesis is falsified
  cheaply and Mechanism C is the next arm.
- **Known approximations, ledgered.** (a) The MP edges are *stateless* across
  blocks (only node latent carries between them, transformed by the intervening
  attention+FFN) — a deliberate simplification of MGN's inter-step edge latent.
  (b) At `frames_per_call>1` the graph is built at `x_t` only, so it bakes
  `x_t` geometry into all k bundled frames (fine at k=1; flagged if composed
  with the k-frames axis). (c) The radius is a per-benchmark constant whose
  off-grid transfer is itself untested — edge-degree histograms on the probe
  case are a build-time diagnostic.
- Orthogonal to and composable with the SAROS/φ-conditioning direction
  (which changes *how the global slices are formed*); this branch adds a
  *parallel local track*.
