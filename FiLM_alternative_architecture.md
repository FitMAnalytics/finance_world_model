# FiLM Alternative: State-Conditional (Hypernetwork) FiLM

**Status**: Deferred idea, not yet implemented. Captured here so it isn't lost after shot 1.

## Motivation

In the current `FiLMWorldModel`, the modulation coefficients `(γ, β)` come from *static* conditioning signals:

- `film_anchor` uses `anchor_state` (raw absolute levels on the lookback's last day)
- `film_event` uses `e_global` (masked mean over event date vectors)

Neither depends on the actual market trajectory inside the window. That means FiLM can encode "oil is at $120 and there was an OPEC cut" regimes, but it *cannot* encode "the last two days were unusually volatile" or "we just had a −3σ drop followed by a flat day" — those are properties of the delta sequence itself, not of the anchor or the events.

The proposal: make FiLM's `(γ, β)` depend on a flattened embedding of the lookback window, combined with the existing static conditioning. This turns FiLM into a small hypernetwork / SE-block and injects window-level context into the modulation without disturbing the rest of the architecture.

## Architecture

Keep the current sequence-based market pathway (per-day `MarketMLP` → `(B, 10, d)` → cross-attn → transformer → head). Only change how `(γ, β)` are produced.

```
flat_state = market_deltas.flatten(start_dim=1)          # (B, 460)

# --- film_anchor (stateful) ---
ctx1 = concat([flat_state, anchor_state], dim=-1)        # (B, 460 + 46)
cond1 = cond_mlp_1(ctx1)                                 # (B, hidden)
γ1, β1 = gamma_head_1(cond1), beta_head_1(cond1)         # (B, d) each

# --- film_event (stateful) ---
ctx2 = concat([flat_state, e_global], dim=-1)            # (B, 460 + d)
cond2 = cond_mlp_2(ctx2)                                 # (B, hidden)
γ2, β2 = gamma_head_2(cond2), beta_head_2(cond2)         # (B, d) each
```

The `γ` and `β` are still broadcast uniformly across the 10 timesteps — so the sequence structure, positional encoding, cross-attention, and temporal transformer all remain valid and unchanged.

## Why this is defensible

- **Well-known pattern**. This is a hypernetwork (Ha et al. 2016), structurally identical to Squeeze-and-Excitation blocks, conditional layer norms used in diffusion / TTS, and gMLP-style gating.
- **Gradient flow is trivial**. Autograd sees a fully differentiable graph through `cond_mlp → gamma_head → γ * h`. Standard backprop; no special tricks.
- **Division of labor**. FiLM still does *uniform* modulation across time — it's the "global context lens." The `TemporalTransformer` still does per-timestep mixing. The hypernetwork just makes the global lens data-aware.
- **Minimal blast radius**. Only `FiLMLayer.__init__` and `forward` change. The rest of the model is untouched.

## What it gains

- Context like "this window is volatile" can now bias the modulation, not just the anchor levels.
- Interaction terms between state and anchor become expressible: the same anchor can produce different `(γ, β)` depending on how the deltas look.

## What it does NOT gain

- FiLM is still uniform across the 10 timesteps. The "bounce-back" pattern — day 8 should mean-revert because day 7 dropped — is still *not* expressible by FiLM. That reasoning must live in the `TemporalTransformer` downstream.
- The hypernetwork modulates the global lens; it does not give FiLM any per-day selectivity.

## Implementation sketch

```python
class StatefulFiLMLayer(nn.Module):
    def __init__(self, cond_flat_dim, cond_static_dim, feature_dim, hidden=64):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_flat_dim + cond_static_dim, hidden),
            nn.GELU(),
        )
        self.gamma_linear = nn.Linear(hidden, feature_dim)
        self.beta_linear  = nn.Linear(hidden, feature_dim)
        # Initialize to identity: (γ, β) = (1, 0) at step 0
        nn.init.zeros_(self.gamma_linear.weight)
        nn.init.ones_(self.gamma_linear.bias)
        nn.init.zeros_(self.beta_linear.weight)
        nn.init.zeros_(self.beta_linear.bias)

    def forward(self, x, flat_state, static_cond):
        # x: (B, 10, d), flat_state: (B, 460), static_cond: (B, cond_static_dim)
        ctx = torch.cat([flat_state, static_cond], dim=-1)
        h = self.cond_mlp(ctx)
        gamma = self.gamma_linear(h).unsqueeze(1)   # (B, 1, d)
        beta  = self.beta_linear(h).unsqueeze(1)    # (B, 1, d)
        return gamma * x + beta
```

Usage in the model's forward:

```python
flat = market_deltas.flatten(start_dim=1)
h = self.market_mlp(market_deltas)           # (B, 10, d) — unchanged
h = self.film_anchor(h, flat, anchor_state)  # stateful
h = self.film_event(h, flat, e_global)       # stateful
# cross_attn, temporal_transformer, head — all unchanged
```

## Things to watch

**Init to identity**. The zero-weight / one-bias init on `gamma_linear` and `beta_linear` ensures that at step 0, FiLM is an identity map regardless of what `cond_mlp` outputs. This matches the current FiLM init and makes early training stable — the hypernetwork only "turns on" gradually as the loss pushes it.

**Parameter budget**. With `hidden = 64` and `d_model = 256`:
- `cond_mlp_1`: `Linear(506, 64) ≈ 32K`
- `gamma_head_1 + beta_head_1`: `2 × 64 × 256 ≈ 33K`
- Same again for `film_event` with slightly different input dim.

Total extra: roughly 130K parameters. Modest relative to the current model (~700K).

Using a wider hidden (e.g., 256) or stacking two layers in `cond_mlp` can push this to 300–500K. On 340 training samples, start narrow.

**Redundancy with the transformer**. Both the hypernetwork and the transformer see the whole window. Two paths for the same information is usually mild — sometimes an ensemble effect helps, sometimes they fight. On small data I'd bet it helps, but monitor for instability.

**Overfitting**. Any increase in parameters needs matching regularization on this dataset size. Add dropout inside `cond_mlp` and consider weight decay bump.

## When to try this

After shot 1 (flat MLP, no transformer) establishes a baseline. If shot 1 beats the current design, the flat-MLP inductive bias is clearly better on this data and the hypernetwork idea has less to prove. If shot 1 does *not* beat the current design, then the sequence + transformer structure is providing value, and adding state-conditional FiLM on top is the cleanest next experiment to enrich that design.

## Related ideas not explored here

- **Per-timestep FiLM**: produce `(γ_t, β_t)` that vary across the 10 timesteps. Breaks FiLM's cheap "global lens" framing and becomes closer to a learned gating network. More capacity, less inductive bias.
- **Hypernetwork for the prediction head** instead of FiLM: use `flat_state` to generate the weights of the output linear layer. Even more hypernetwork-y; adds more parameters; harder to train.
- **MLP-Mixer-style token mixing** before the current `MarketMLP`: a `Linear(10, 10)` applied along the time axis gives first-order cross-day interactions with ~100 parameters, then the rest of the pipeline is unchanged. A lighter alternative to full flattening that preserves the sequence.
