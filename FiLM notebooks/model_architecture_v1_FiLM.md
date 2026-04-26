# Oil Price World Model — Architecture Design Document

## 1. Motivation and Context

We are building a world model that predicts future oil market states from historical market data and exogenous geopolitical/macroeconomic events. The model takes as input a window of recent market performance plus relevant events (sourced from GDELT, enriched via Gemini), and outputs predicted market trajectories.

The key architectural insight driving this design is **causal asymmetry**: events (geopolitical shocks, policy changes, supply disruptions) cause changes in market state, but market state does not cause events. The architecture should encode this causal direction explicitly rather than hoping a model discovers it from concatenated features.

---

## 2. Input Representations

### 2.1 Market State: Daily Changes (Delta Encoding)

Rather than feeding raw market values (which are non-stationary and scale-dependent), we encode the market as **daily log returns**:

```
δ_t = log(p_t / p_{t-1})
```

for each of the ~50 features (commodity prices, futures, macro indicators, etc.) at each timestep.

**Input shape**: A sequence of 10 timesteps, each a 50-dimensional vector of daily deltas.

```
market_deltas: (batch, 10, number of features)
```

### 2.2 Anchor State: Current Absolute Levels

The delta sequence tells you the trajectory shape, but not where in absolute space you are. A +3% oil move means something different at $40 (pandemic lows) vs $120 (supply crisis). We preserve this context as a single **anchor state vector** — the most recent day's raw feature values.

```
anchor_state: (batch, number of features)
```

This enters the architecture via FiLM modulation (Section 3.2), not concatenation.

### 2.3 Events: Date-Level Text with Market Context

Events are natural language descriptions of geopolitical/macroeconomic occurrences sourced from GDELT (top-200 events per year by composite impact score, filtered to market-relevant events via a Qwen-based classifier, enriched via Gemini with Google Search grounding).

Rather than encoding each event independently, we format **one text per calendar date** that bundles (a) the full market context for that day and (b) every market-relevant event on that date. Dates with no events still get a text (market context only), so the encoder sees a continuous daily stream.

```
"Market on 2015-05-22: WTI $58.72, Brent $65.37, Gold $1204.50, ... DFF 0.12%, DGS10 2.21% ...
Events (2015-05-22):
- Thousands of Shia militia fighters arrived at ...
- OPEC announces ..."
```

Dates without events end with `"No significant events."` instead of an event block.

**Why prepend market context to the text:**
- The LLM can interpret event significance conditionally on market state. "Production cut" means something different when oil is at $40 vs $120.
- This does NOT create a circular dependency: the market encoder processes deltas independently; the LLM processes text (which happens to mention current levels) independently. They merge later via modulation.
- Natural language is a lightweight bridge — the LLM already understands numeric context in text form.

**Date-level pooling**: Each date-text is passed through Qwen3-4B once, and a single pooled vector (`qwen_hidden = 2560`) is extracted for that date. A lookback window of `L = 10` days collects up to `max_events_per_window = 8` such date vectors, padded/masked as needed.

---

## 3. Architecture (Shot 1: Flat-State FiLM)

### 3.1 Overview

Shot 1 deliberately uses a **flat-state** market pathway. The full `(10, 46)` lookback window is flattened and embedded by a 2-layer MLP, so cross-day and cross-feature interactions are captured directly in the first linear layer of the state embedding. With no sequence axis, the original cross-attention and temporal transformer blocks are removed — they have no sequence to operate over, and any "global context" role they played is now absorbed by the flat MLP plus the two FiLM layers. See `20260411_architecture_discussion.md` for the reasoning and `FiLM_alternative_architecture.md` for a deferred stateful-FiLM variant that re-introduces the sequence pathway.

```
┌───────────────────────────────────┐     ┌──────────────────────────────────────┐
│     MARKET PATHWAY                │     │     EVENT PATHWAY                    │
│                                   │     │                                      │
│  market_deltas (10 × 46)          │     │  date_texts (L strings, market       │
│         │                         │     │  context + events per calendar day)  │
│     flatten → (460,)              │     │         │                            │
│         │                         │     │    Qwen3-4B (Stage 1: cached;        │
│     Linear(460, H) + GELU         │     │    Stage 2: 4-bit + LoRA)            │
│     Dropout                       │     │         │                            │
│     Linear(H, d) + GELU           │     │    pool last non-pad token → (2560,) │
│         │                         │     │         │                            │
│     state_emb (d,)                │     │    Linear_proj → (K × d)             │
│         │                         │     │         │                            │
│  ┌──────┴──────┐                  │     │    ┌────┴────┐                       │
│  │  FiLM       │◄── anchor_s₀     │     │    │ Masked  │                       │
│  │  (anchor)   │    (46 → γ,β)    │     │    │ Mean    │──► e_global (d)       │
│  └──────┬──────┘                  │     │    └────┬────┘                       │
│         │                         │     │         │                            │
│  ┌──────┴──────┐                  │     │         │                            │
│  │  FiLM       │◄──────────────────────── e_global                             │
│  │  (event)    │                  │     │                                      │
│  └──────┬──────┘                  │     └──────────────────────────────────────┘
│         │                         │
│     h (d,)                        │
│         │                         │
│  Linear(d, H × 46) → reshape      │
│         │                         │
│  predicted_deltas (10 × 46)       │
└───────────────────────────────────┘
```

### 3.2 Market State Embedding

The market pathway is a **flat MLP**, not a per-day sequence encoder. The full lookback window is flattened and passed through a 2-layer MLP:

```python
flat = market_deltas.flatten(start_dim=1)              # (B, L × n_features) = (B, 460)
h = GELU(Linear(flat, state_mlp_hidden))               # (B, 460) → (B, 512)
h = Dropout(h)
state_emb = GELU(Linear(h, d_model))                   # (B, 512) → (B, 256)
```

With this design, the first linear layer has direct access to every (day, feature) input, so it can express **any cross-day, cross-feature interaction** — exactly what we want for momentum, mean reversion, and lead-lag patterns across commodities.

Output: `(batch, d_model=256)` — a single flat state vector, not a sequence.

**Why flat instead of sequence + transformer**: with `L = 10` (fixed, small) and only a few hundred training samples, a flat MLP has a better inductive-bias-to-capacity ratio than a per-day embedding feeding a transformer. Self-attention's `O(L²)` machinery is overkill for `L = 10`, and the shared-per-day embedding cannot express "yesterday down → today bounce" at all — it has no mechanism for cross-day interaction. The flat MLP captures all such interactions explicitly in its first layer. This aligns with findings from Zeng et al. 2023 ("Are Transformers Effective for Time Series Forecasting?") that simple linear/MLP models match or beat transformer forecasters on short horizons.

### 3.3 FiLM Layer 1: Anchor State Modulation

The anchor state s₀ (current absolute market levels on the last day of the lookback) modulates the flat state embedding via Feature-wise Linear Modulation:

```python
γ₁ = Linear_γ1(s₀)        # (46,) → (d,)
β₁ = Linear_β1(s₀)        # (46,) → (d,)
state_emb ← γ₁ ⊙ state_emb + β₁    # element-wise on a (d,) vector
```

**Semantics**: The anchor state sets the "regime lens" through which the embedded delta window is interpreted. When oil is at $40, certain delta patterns mean one thing; at $120, they mean another. FiLM's multiplicative interaction (γ) captures this context-dependent reweighting more naturally than additive concatenation.

**Identity init**: the FiLM linears are initialized so that `(γ, β) = (1, 0)` at step 0, making the layer an identity map until the loss pushes it otherwise. This matches the original FiLM paper's recommendation and keeps early training stable.

### 3.4 Event Encoder (Single-Vector Date Embeddings)

**Backbone**: Qwen3-4B (hidden size 2560). In Stage 1 it is used purely as a frozen feature extractor via pre-computed embeddings; in Stage 2 it is loaded 4-bit quantized with LoRA adapters (QLoRA) and fine-tuned live.

**Input**: Each date-level text (Section 2.3) — one string per calendar day in the lookback window.

**Output extraction**: For each date text, run Qwen3-4B and take a single pooled hidden state of dimension `qwen_hidden = 2560`. This yields one vector per date, not a token sequence — the single-vector formulation avoids variable-length token padding and keeps Stage 1 training fast.

```python
date_vec = qwen_pool(date_text)          # (2560,)
event_repr = Linear_proj(date_vec)       # (2560,) → (d,)
```

For a lookback window of L = 10 days, at most `max_events_per_window = 8` date vectors are collected (older dates first) and padded with a boolean mask. Each row of `event_embeds` is one date, not one individual event.

**Global event pool**: Masked mean across the valid date vectors produces a single summary vector:

```python
e_global = masked_mean(event_reprs, event_mask)    # (K, d) → (d,)
```

If the window contains no valid date vectors, `e_global` falls back to a learned `no_event_embed` parameter.

**Stage 1 vs Stage 2**:
- **Stage 1** (`02_stage1_single_vector.ipynb`): Date vectors are read from a cached `event_embeddings.pt` file produced by `01_precompute_event_embeddings.ipynb`. No LLM is loaded during training — only the small FiLM model runs on GPU.
- **Stage 2** (`03_stage2_qlora.ipynb`): Qwen3-4B is loaded in 4-bit (nf4, double-quant, bf16 compute) with LoRA adapters on `q_proj` / `v_proj`. Texts are tokenized and encoded live inside the forward pass, deduplicated across the batch, and the last non-padding hidden state is used as the pooled vector. The Stage 1 projection weights are transferred at initialization.

### 3.5 FiLM Layer 2: Event Regime Modulation

The global event pool modulates the (already anchor-modulated) flat state, encoding "what kind of geopolitical regime are we in":

```python
γ₂ = Linear_γ2(e_global)    # (d,) → (d,)
β₂ = Linear_β2(e_global)    # (d,) → (d,)
h ← γ₂ ⊙ h + β₂             # element-wise on a (d,) vector
```

**Semantics**: This captures coarse regime shifts — "we're in a supply disruption world" vs "we're in a demand-driven rally" vs "geopolitical calm." The same state embedding should be interpreted differently depending on the event regime.

**Why no cross-attention anymore**: in the original design, cross-attention let *individual* market timesteps attend to specific events (day 3 attending heavily to an OPEC event, day 8 less). With a flat state embedding there are no per-day queries to issue, so cross-attention would reduce to a single-query weighted sum — essentially a re-derivation of `e_global` with query-conditional weights, almost entirely redundant with the `film_event` path above. Removed for shot 1.

### 3.6 Prediction Head

A single linear layer maps the flat state vector to a flattened `(horizon × n_features)` forecast, then reshapes:

```python
predicted = Linear(h).view(B, horizon, n_features)    # (B, d) → (B, 10, 46)
```

**Why a flat projection**: with a flat state, there's no per-day slot to decode from. The head has to produce all 10 future days at once, and it does so by learning 10 × 46 = 460 independent readouts from the same `d`-dim state. This is the standard head used by DLinear/NLinear forecasters.

**Loss**: MSE on predicted deltas, masked to the 6 critical commodity price features (WTI, Brent, Gold, Natural Gas, Silver, Copper). Masking prevents the model from spending capacity on noisy, unpredictable features.

```python
loss = MSE(predicted_deltas[:, :, critical_features],
           target_deltas[:, :, critical_features])
```

### 3.7 What was removed vs. the original design

For historical context and to make it easy to reintroduce later, here's what shot 1 drops relative to the original sequence-based design:

| Module | Status | Reason |
|---|---|---|
| Per-day `MarketMLP` (shared weights across timesteps) | **Replaced** with flat `MarketStateMLP` | Shared per-day projection cannot express cross-day interactions; flat MLP can. |
| `EventMarketCrossAttention` (market Q attends events K/V) | **Removed** | With a flat state there's no per-day query; degrades to a redundant re-pooling of events. |
| `TemporalTransformer` (self-attention across 10 days) | **Removed** | No sequence axis remains for self-attention to operate on. |
| FiLM `(γ, β)` broadcast across 10 timesteps | **Removed** | Modulation now applied directly element-wise on the flat state vector. |

The two FiLM layers, the event encoder, and the anchor-state / event-regime conditioning all remain — they are the defining features of this architecture and the parts that distinguish it from a generic tabular MLP forecaster.

---

## 4. Training Strategy

### 4.1 Staged Training

Training is split across four notebooks to avoid GPU OOM (Qwen and the training model cannot co-reside on a Kaggle T4) and to separate concerns:

**Notebook 01 — `01_precompute_event_embeddings.ipynb`**:
- Loads Qwen3-4B (bf16, full precision) once.
- Builds one date-level text per calendar day in 2015–2016 (market context + that day's market-relevant events).
- Extracts a Qwen hidden-state representation per date and saves `event_embeddings.pt` keyed by `YYYY-MM-DD`.
- Runs on GPU with Qwen loaded; no training.

**Notebook 02 — `02_stage1_single_vector.ipynb` (Stage 1)**:
- No LLM loaded. Reads cached date vectors.
- Trains the entire flat-state FiLM world model: `MarketStateMLP`, both FiLM layers, `EventEncoder` projection + `no_event_embed`, and `PredictionHead`.
- Optimizer: AdamW, lr `1e-3`, cosine schedule, 150 epochs, batch size 64.
- Loss: MSE on the 6 critical commodity price features.
- Saves `stage1_best.pt`.
- Purpose: Establish stable market dynamics modeling against fixed event representations.

**Notebook 03 — `03_stage2_qlora.ipynb` (Stage 2)**:
- Loads Qwen3-4B in 4-bit (QLoRA: nf4, double-quant, bf16 compute) with LoRA adapters (`r=16`, `alpha=32`, targets `q_proj` / `v_proj`), enabling gradient checkpointing.
- Transfers Stage 1 weights into the Stage 2 model (`MarketStateMLP`, both FiLM layers, `PredictionHead`, plus the event-encoder projection + `no_event_embed`).
- Dataset returns raw event **texts**; Qwen encodes them live in the forward pass, deduplicating repeated texts across the batch.
- Two-group AdamW: market pathway at `1e-4`, LoRA at `1e-5` (10× lower). 30 epochs, batch size 4, grad accumulation 4.
- Saves `stage2_best.pt`.
- Purpose: Let the event encoder adapt its representations to what market prediction actually needs, without the VRAM cost of a full-precision Qwen.

**Note**: Notebooks 03 and 04 still contain the original sequence + cross-attention + transformer architecture. They will need to be synced to the shot-1 flat-state model class before Stage 2 / evaluation can run against a Stage 1 checkpoint produced by the updated notebook 02. This is a follow-up TODO, not required for shot-1 Stage 1.

**Notebook 04 — `04_evaluate_2016.ipynb`**:
- Reconstructs the Stage 2 model (same QLoRA config) and loads `stage2_best.pt`.
- Runs day-by-day over 2016, predicts 10-day log-return trajectories, recovers prices via `p_last * exp(cumsum(log_returns))`, and reports per-feature MSPE / RMSPE with a focus on the critical features.

### 4.2 Single-vector vs cached-token-sequence variants

The repo contains two Stage 1 notebooks:

- `02_stage1_single_vector.ipynb` — **current default**. One pooled vector per date. Faster, fixed-shape batches, no token-length concerns.
- `02_stage1_cached_embeddings.ipynb` — kept for reference. Consumes full token sequences per date and concatenates them across the lookback window up to `max_event_tokens = 512`. More expressive but slower and padding-heavy.

All downstream notebooks (03, 04) assume the single-vector formulation.

Notebook 01 pools each date-level text to a single `(1, hidden)` vector by taking the last non-padding token's hidden state from Qwen3-4B — standard causal-LM pooling — so the saved cache drops straight into the single-vector Stage 1 dataset without any shape reinterpretation.