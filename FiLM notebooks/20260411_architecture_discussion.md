# 2026-04-11 Architecture Discussion

A working log of the architectural debate that led to the shot-1 flat-state FiLM design, captured while the reasoning is still fresh. Two related but distinct concerns drove the conversation:

1. Does the current per-day `MarketMLP` + FiLM + transformer pathway actually capture cross-day interactions like "yesterday down → today bounce"?
2. If not, what's the minimal change to fix it, and what's the more conservative alternative?

The conclusion was: Option 1 (flatten the state, drop the now-unused attention layers) for shot 1, with Option 2 (state-conditional FiLM / hypernetwork) captured in `FiLM_alternative_architecture.md` as a deferred experiment.

## The core concern

The original `FiLMWorldModel` market pathway is:

```
market_deltas (10, 46) ─► MarketMLP ─► (10, 256)
                          └─ Linear(46, 256) + GELU, shared across the 10 days
                          ► FiLM_anchor ─► (10, 256)   # uniform (γ, β) across days
                          ► FiLM_event  ─► (10, 256)   # uniform (γ, β) across days
                          ► CrossAttn   ─► (10, 256)   # market Q, event K/V
                          ► TemporalTransformer ─► (10, 256)   # self-attn across days
                          ► PredictionHead per position ─► (10, 46)
```

At no point before the `TemporalTransformer` do the 10 days of market data "talk to each other." Specifically:

- **MarketMLP** is `Linear(46, 256) + GELU` applied along the last axis. PyTorch broadcasts this across the time dimension, so each day gets embedded independently with shared weights. No cross-day mixing.
- **FiLM** broadcasts the same `(γ, β)` to all 10 timesteps. It cannot treat day 3 differently from day 8. The conditioning (`anchor_state`, `e_global`) is time-invariant by construction.
- **Cross-attention** uses market timesteps as queries and events as K/V. Market queries don't attend to each other, so this also doesn't mix days.

That means **any temporal interaction between the 10 lookback days has to be learned entirely by the 2-layer `TemporalTransformer`**, which has moderate capacity and has to compete with the cross-attention and FiLM for gradient signal on a tiny training set (~340 samples in 2015).

The concrete scenario that crystallized the concern: "if the market dropped a lot yesterday, today there should often be a bounce-back." FiLM cannot express this, because its modulation is uniform across time — it can say "the whole window should be interpreted through a mean-reverting lens" (static context), but not "apply a mean-reverting correction specifically to day *t+1* conditional on day *t* being a drop." That latter kind of reasoning is strictly cross-day and only the transformer can do it in the original design.

## Two ways to fix it

### Option 1: flatten the state

Change the market pathway so the first linear layer sees all 10 days at once:

```
flatten (460,) ─► Linear(460, 512) + GELU
                ─► Dropout
                ─► Linear(512, 256) + GELU
                ─► state_emb (256,)
```

Because the first linear layer has `460 × 512 ≈ 235K` weights and every output unit has a direct weight on every input (day, feature) pair, the network can express any pairwise cross-day, cross-feature interaction in a single linear step. This is the tabular-MLP view of time-series forecasting: treat the `(10, 46)` window as a 460-dim feature vector and let a fully-connected net do the rest.

Immediate consequences once the state is flat:

- **FiLM still works** — just element-wise modulation on a `(256,)` vector, no broadcast needed.
- **Cross-attention** has only a single query (the flat state). It reduces to a single weighted sum of events using one query — basically a re-derivation of `e_global` with query-conditioning. It's almost entirely redundant with the `film_event` pathway that already uses `e_global`.
- **Temporal transformer** operates on sequences. With no sequence axis, self-attention over "length 1" collapses to a linear-plus-LayerNorm — pure dead weight.

So the clean version of Option 1 removes both cross-attention and the transformer, leaving:

```
market_deltas → MarketStateMLP → FiLM_anchor → FiLM_event → PredictionHead
```

That's the shot-1 architecture.

**Why this might even be *better*** (not just simpler): Zeng et al. 2023 ("Are Transformers Effective for Time Series Forecasting?") showed that trivially simple linear/MLP forecasters match or beat transformer architectures on standard time-series benchmarks, especially at short horizons. For `L = 10`, `H = 10`, ~340 training samples, the transformer's inductive bias is probably not earning its keep, and the flat MLP's direct cross-day pairwise access is probably better matched to the signal structure.

**Risks**:
- More parameters in the market pathway (flat MLP is ~420K vs the per-day MLP's ~12K + transformer's ~530K). Net change is smaller than it looks once the transformer is gone.
- Tied to a fixed `L = 10`. Different lookback lengths require retraining from scratch. Fine for now.
- Fully connected, no weight sharing across days — less data-efficient in principle. Dropout + weight decay help.

### Option 2: state-conditional FiLM (hypernetwork)

Keep the sequence-based architecture (per-day `MarketMLP`, cross-attention, transformer — everything). Only change how `(γ, β)` are generated for the two FiLM layers: feed them a flattened embedding of the lookback window in addition to the current static conditioning.

```python
flat = market_deltas.flatten(start_dim=1)              # (B, 460)
ctx1 = cat([flat, anchor_state], -1)                   # (B, 506)
cond1 = cond_mlp_1(ctx1)                               # (B, hidden)
γ₁, β₁ = gamma_head_1(cond1), beta_head_1(cond1)       # (B, d)

ctx2 = cat([flat, e_global], -1)                       # (B, 460 + d)
cond2 = cond_mlp_2(ctx2)                               # (B, hidden)
γ₂, β₂ = gamma_head_2(cond2), beta_head_2(cond2)       # (B, d)
```

Then FiLM broadcasts `(γ, β)` across the 10 timesteps as before. The sequence, cross-attention, and transformer are all untouched.

**What this is**: a small hypernetwork. Structurally identical to Squeeze-and-Excitation blocks, conditional LayerNorm (used in diffusion / TTS), gMLP gating, and the original FiLM paper's own examples of learned conditioning. Nothing exotic.

**What it gains**: FiLM is no longer purely regime-driven. It can now say things like "this lookback window is volatile" or "the same $120 anchor *plus* two recent red days means something different than the same $120 anchor alone" — window-aware global modulation.

**What it still cannot do**: FiLM's `(γ, β)` are still broadcast uniformly across time. Per-day differential reasoning ("day 8 should bounce because day 7 dropped") still has to live in the `TemporalTransformer`. Option 2 makes FiLM *smarter*, not *time-aware*.

**Training stability**: as long as `gamma_head` and `beta_head` are initialized so that `(γ, β) = (1, 0)` at step 0, the whole FiLM layer starts as an identity map and gradually "turns on" as the loss trains `cond_mlp`. Standard hypernetwork init.

**Gradient flow**: trivially fine. Autograd sees a differentiable path through `cond_mlp → gamma_head → γ * h` and treats it like any other computation. No special training tricks.

## Why Option 1 for shot 1

Option 1 is the more aggressive structural change, but it's *simpler* — it removes two whole modules and leaves a cleaner forward pass. Option 2 is more conservative w.r.t. the existing design but *adds* complexity (a hypernetwork on top of an already large pipeline).

For a shot-1 baseline, the priorities are:

1. **Something that runs end-to-end quickly** → flat MLP beats "existing complex pathway + extra hypernetwork on top."
2. **Inductive bias matched to the problem size** → `L = 10`, ~340 samples, flat MLP is probably the right scale.
3. **Easy to reason about** → fewer modules, shorter forward pass.
4. **A clean baseline against which to measure future ideas** → if shot 1 works, Option 2 becomes an additive experiment on top. If shot 1 underperforms, the sequence-based design's inductive bias was actually earning its keep and Option 2 is the right next step.

Option 1 also lets us confirm the *data* is good. With all the attention machinery gone, any remaining failures are much more directly attributable to the input representation or the event embeddings, not the architecture.

## Dimensions chosen for shot 1

```
MarketStateMLP:
    Linear(460, 512) + GELU + Dropout(0.1)
    Linear(512, 256) + GELU
    
FiLM_anchor:  Linear(46, 256), Linear(46, 256)     # identity init
FiLM_event:   Linear(256, 256), Linear(256, 256)   # identity init

PredictionHead:
    Linear(256, 460) → reshape (10, 46)
```

Rationale:

- `state_mlp_hidden = 512`: roughly 2× the flat input dim, a standard "widen in the middle" MLP. Wider (e.g., 1024) is plausible but risks overfit on 340 samples; can revisit if shot 1 underfits.
- `d_model = 256`: unchanged from the current design, so FiLM dimensions and event projection stay consistent.
- Dropout only between the two MLP layers. FiLM layers start at identity, so they effectively have "built-in" regularization early in training.
- No layer norm in the MLP. Adding one is a common next tweak if training is unstable; unnecessary to start.

## Open questions

- Does shot 1 even need the event pathway for Stage 1 training, or can we ablate it? Easy A/B test: run Stage 1 with and without `film_event` and see what the critical-feature MSE does.
- Should we hold out 2015 data for a validation loss instead of only tracking train loss? Right now `02_stage1_single_vector.ipynb` saves the best *train* loss checkpoint, which invites overfit. A small validation split would be a cheap improvement.
- Is ~340 samples really enough to train a 420K-param MLP without severe overfit? The existing per-day design has similar param count, so empirically yes — but this is the biggest risk to watch.
- When Option 2 gets its turn, do we stack it on top of shot 1 (flat state + hypernetwork FiLM) or on top of the original sequence design? Stacking on shot 1 is simpler and still exercises the hypernetwork idea. Stacking on the original is a bigger pivot but tests whether the sequence structure was the thing we were missing.

## Todo after shot 1

- Sync `03_stage2_qlora.ipynb` and `04_evaluate_2016.ipynb` to the shot-1 model class so Stage 2 / evaluation can run against a Stage 1 checkpoint. Currently they still define the old sequence + cross-attention + transformer architecture and will fail to load the new `stage1_best.pt`.
- Add a validation split to Stage 1 training (e.g., last 30 days of 2015) so the best checkpoint is selected on val loss rather than train loss.
- Consider the ablations: `film_event` on/off, `film_anchor` on/off, state MLP width sweep.
- If shot 1 beats the original design: treat Option 2 as the next experiment.
- If shot 1 underperforms the original: either the sequence/transformer was doing useful work after all (bring it back, consider Option 2 on top) or the bottleneck is in the event encoder / data (investigate those independently).
