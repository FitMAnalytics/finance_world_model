# Event Data Pipeline Plan

Building event data for a volatility-prediction world model. Input: news headlines + first paragraphs from NYT and Guardian free APIs. Output: structured event records suitable as LLM-context input to a downstream world model.

## Goal

Produce a clean, structured event stream where each record is a market-relevant event with verified context. These records will feed a world model that takes (market data + recent events) → probability of a high-volatility / black-swan regime.

Key property of the target data: **bias toward recall over precision**. Volatility-relevant signals are often noisy, clustered, or seemingly minor in isolation. Over-filtering destroys signal we can't get back.

## Pipeline Overview

```
Raw headlines (NYT + Guardian)
    ↓
[Stage 1] LLM one-shot filter       → runs on Kaggle (16GB VRAM)
    ↓
[Stage 2] Function-calling search    → runs on local machine (8GB VRAM)
    ↓
Structured event records
```

Two distinct stages, run on different hardware, optimized for different constraints.

---

## Stage 1: Filter (Kaggle, 16GB VRAM)

### Purpose

Drop clearly market-irrelevant headlines. Keep anything that could plausibly be relevant to market volatility. This is a cheap, high-recall classification pass — not a judgment of "importance."

### Approach

One-shot prompt to an instruct LLM. For each headline + first paragraph, output a structured JSON verdict:

```
{
  "is_potentially_market_relevant": bool,
  "category": ["monetary_policy" | "geopolitical" | "financial_stress"
               | "sovereign_stress" | "commodity_shock" | "corporate_stress"
               | "policy_shock" | "disaster_health" | "other" | "none"],
  "known_to_model": "well-known" | "partially-known" | "unknown",
  "confidence": 0.0 - 1.0
}
```

Filtering rule: **keep** anything where `is_potentially_market_relevant` is true OR `confidence` is below a threshold (e.g., 0.7). When uncertain, keep it. Stage 2 will do the expensive verification.

### Model choice

With 16GB VRAM on Kaggle (T4 × 2 = 2 × 16GB, or a single P100), target something in the 8B–14B range at 4-bit quantization:

- **Qwen3-8B or Qwen3-14B** (4-bit via bitsandbytes) — already in the Qwen ecosystem, solid JSON output
- **Llama 3.1 8B Instruct** (4-bit) — reliable alternative

14B at 4-bit fits in ~10GB, leaves room for KV cache and batching. Prefer the larger model for this stage since category judgment benefits from stronger reasoning, and filter mistakes propagate downstream.

### Implementation notes

- Batch headlines aggressively; this stage is throughput-bound, not latency-bound
- Use constrained JSON decoding (outlines / XGrammar / Qwen's structured output) to guarantee parseable output
- Consider a cheap regex/keyword pre-filter before the LLM call to drop obvious non-candidates (sports, entertainment, local crime) and save compute
- Define categories with 2-3 concrete examples each in the prompt — category labels without examples produce inconsistent classifications

### What this stage does NOT do

- Does not judge "importance" — only category membership and relevance
- Does not expand or enrich — that's Stage 2
- Does not verify facts — that's Stage 2

---

## Stage 2: Expand & Verify (Local machine, 8GB VRAM)

### Purpose

For each survivor from Stage 1, verify the event against online sources and produce a structured event record with enriched context.

### Approach: Fixed pipeline, NOT an agentic loop

No agentic loop at the start. The workflow is deterministic:

```
Filtered headline + first paragraph + Stage 1 metadata
    ↓
LLM call 1: generate 1–2 search queries
    ↓
Search tool (Serper / Brave / Google Custom Search)
    ↓
Optionally fetch top 2–3 result URLs
    ↓
LLM call 2: synthesize structured event JSON from headline + search results
    ↓
Event record
```

Two LLM calls, one tool call per event. Predictable, parallelizable, easy to debug.

### Model choice

With 8GB VRAM (RTX 2070), need tight constraints:

- **Qwen3-4B** (4-bit) — ~3GB, leaves headroom for context and tool-call tokens
- **Qwen3-8B** (4-bit, tight) — ~5GB, feasible but limits context length
- Recommended starting point: **Qwen3-4B at 4-bit**. Already familiar, strong native tool-calling, room for long-ish context when ingesting search results. Move up to 8B only if 4B's expansion quality is insufficient.

### Output schema (target)

Each event record should include at minimum:

```
{
  "event_id": str,
  "date": iso8601,
  "headline": str,
  "category": [str],
  "summary": str,              # LLM-written, grounded in retrieved sources
  "key_entities": [str],       # orgs, people, countries
  "affected_regions": [str],
  "affected_assets": [str],    # commodities, asset classes, sectors
  "surprise_level": "expected" | "partially_expected" | "surprise" | "unknown",
  "scope": "local" | "regional" | "global",
  "sources": [url],
  "confidence": 0.0 - 1.0
}
```

Fields like `surprise_level` and `scope` are specifically chosen for volatility-relevance — vol is driven by surprises and scope, not by whether something is "big news."

### Search tool options

- **Serper** (~$50/mo for 50k queries) — clean API, recommended if volume is high
- **Brave Search API** — reasonable free tier
- **Google Custom Search** — free 100/day, fine for a one-time dataset build
- **DuckDuckGo (`duckduckgo-search`)** — free but flaky and rate-limited

### Implementation notes

- Use constrained JSON output for the final synthesis call
- Cache search results on disk keyed by query — reruns during development will be free
- Log the raw search results alongside the structured output, so you can re-synthesize later with a better model without re-paying for search
- Parallelize across events (async tool calls), not within an event
- Run on local machine since tool calls are I/O-bound, not compute-bound — 8GB is fine

---

## Progression / Iteration Plan

1. **v0**: Filter + fixed two-call pipeline. Build first dataset end-to-end.
2. **Review**: Manually inspect a sample (50–100 events). Identify failure modes:
   - Hallucination in synthesis?
   - Missing context the model should have searched for?
   - Entity disambiguation errors?
   - Events that should have been clustered with others?
3. **Targeted fixes**: Add mechanisms (agentic loop, clustering pass, second-pass verifier on low-confidence records) *only* for observed failure patterns.
4. **Scale**: Once the per-event pipeline is trusted, run across the full headline corpus.
