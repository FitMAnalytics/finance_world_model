# %% [code]
# %% [code]
# %% [code]
"""Stage 2 event expansion — Kaggle (T4 x2) fork.

Speed-ups vs. the local script:
  1. fp16 weights (no 4-bit quantization) — faster matmul on T4.
  2. Batched generation for both gen_queries and synthesize calls.
  3. ThreadPool-parallel Wikipedia + Tavily lookups across a batch.
  4. Optional dual-replica mode: load the model once per GPU and round-robin
     batches between them. Set USE_BOTH_GPUS = True if you want ~2x throughput.

Kaggle setup checklist:
  - Accelerator: GPU T4 x2
  - Internet: ON
  - Add-ons > Secrets: TAVILY_API_KEY
  - Add your filtered parquet as a Kaggle dataset and update INPUT_PATH below.

Run from a Kaggle notebook cell:
    !python /kaggle/working/stage2_event_expansion_kaggle.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote

# Silence HF loading spam — must be set BEFORE importing transformers/hub.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Reduce fragmentation under long-running batched generation on T4s.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=".*torch._check_is_size.*")

import pandas as pd
import requests
import torch
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer



# --------------------------------------------------------------------------- #
# CONFIG — edit these before running on Kaggle
# --------------------------------------------------------------------------- #
INPUT_PATH = Path("/kaggle/input/datasets/zygong1994/nyt-filtered-1516/nyt_filtered_201507_201509_pt3.parquet")
OUTPUT_PATH = Path("/kaggle/working/nyt_stage2_201507_201509_pt3.parquet")
# Optional: point at a parquet from a prior Kaggle run (uploaded as a dataset)
# to skip event_ids already done. None = resume only from OUTPUT_PATH within this session.
RESUME_FROM_PATH: Optional[Path] = None
# e.g. Path("/kaggle/input/<YOUR-STAGE2-PARTIAL>/nyt_stage2_<YYYY_MM>.parquet")

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LIMIT: Optional[int] = None         # process only first N rows (None = all)
BATCH_SIZE = 8                      # rows per gen_queries call (short outputs)
SYNTH_BATCH_SIZE = 4                # rows per synthesize call (thinking + 1024 tok is memory-heavy)
SEARCH_WORKERS = 4                 # parallel HTTP calls for Wikipedia/Tavily
RESUME = True                       # skip event_ids already in OUTPUT_PATH
USE_BOTH_GPUS = True                # load one replica per GPU if 2+ GPUs visible

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
tavily_api = user_secrets.get_secret("TAVILY_API_2")
HF_token = user_secrets.get_secret("HF_TOKEN")

login(token=HF_token)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class QueryGenOutput(BaseModel):
    queries: List[str] = Field(..., min_length=1, max_length=2)


class EventRecord(BaseModel):
    event_id: str
    headline: str
    date: str
    event_time: Optional[str]
    category: List[str]
    summary: str
    key_entities: List[str]
    affected_regions: List[str]
    affected_assets: List[str]
    surprise_level: str
    scope: str
    sources: List[str]
    confidence: float


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_TAVILY_URL = "https://api.tavily.com/search"
_UA = {"User-Agent": "finance-world-model/0.1 (research; contact via repo)"}

_search_lock = threading.Lock()
_wiki_calls = 0
_tavily_calls = 0


def _wiki_search(query: str, k: int) -> List[Tuple[str, str, str]]:
    global _wiki_calls
    with _search_lock:
        _wiki_calls += 1
    resp = requests.get(
        _WIKI_SEARCH_URL,
        params={"action": "query", "list": "search", "srsearch": query,
                "srlimit": k, "format": "json"},
        headers=_UA, timeout=10,
    )
    resp.raise_for_status()
    out: List[Tuple[str, str, str]] = []
    for r in resp.json().get("query", {}).get("search", []):
        title = r["title"]
        slug = quote(title.replace(" ", "_"))
        url = f"https://en.wikipedia.org/wiki/{slug}"
        try:
            s = requests.get(_WIKI_SUMMARY_URL + slug, headers=_UA, timeout=10)
            extract = s.json().get("extract", "") if s.status_code == 200 else ""
        except Exception:
            extract = ""
        if not extract:
            extract = re.sub(r"<[^>]+>", "", r.get("snippet", ""))
        out.append((title, extract, url))
    return out


def _tavily_search(query: str, k: int) -> List[Tuple[str, str, str]]:
    api_key = tavily_api
    if not api_key:
        return []
    global _tavily_calls
    with _search_lock:
        _tavily_calls += 1
    resp = requests.post(
        _TAVILY_URL,
        json={
            "api_key": api_key,
            "query": query,
            "max_results": k,
            "search_depth": "basic",
            "include_answer": False,
        },
        headers=_UA, timeout=15,
    )
    if resp.status_code >= 400:
        try:
            body = resp.json()
            msg = body.get("detail") or body.get("error") or body
        except ValueError:
            msg = resp.text[:300]
        raise RuntimeError(f"Tavily {resp.status_code}: {msg!r}")
    return [
        (r.get("title", ""), r.get("content", ""), r.get("url", ""))
        for r in resp.json().get("results", [])
    ]


@lru_cache(maxsize=4096)
def search(query: str, k: int = 3) -> tuple:
    try:
        hits = _wiki_search(query, k)
        if hits:
            return tuple(hits)
    except Exception as e:
        print(f"[search] Wikipedia error: {e}", file=sys.stderr)
    try:
        return tuple(_tavily_search(query, k))
    except Exception as e:
        print(f"[search] Tavily error: {e}", file=sys.stderr)
        return tuple()


# --------------------------------------------------------------------------- #
# Model — fp16, optional dual-replica across GPUs
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

tokenizer = None      # type: ignore[assignment]
_models: list = []    # one entry per replica, each (model, device, lock)
_rr_counter = 0
_rr_lock = threading.Lock()


def _load_one(model_id: str, device: str):
    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    m.eval()
    return m


def load_model(model_id: str) -> None:
    global tokenizer, _models
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched decoding

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        m = _load_one(model_id, "cpu")
        _models.append((m, "cpu", threading.Lock()))
        print(f"[model] {model_id} loaded on CPU")
        return

    devices = [f"cuda:{i}" for i in range(n_gpu)] if USE_BOTH_GPUS else ["cuda:0"]
    for d in devices:
        m = _load_one(model_id, d)
        _models.append((m, d, threading.Lock()))
        alloc_gb = torch.cuda.memory_allocated(int(d.split(":")[1])) / 1024**3
        print(f"[model] {model_id} replica on {d}: {alloc_gb:.2f} GB")


def _pick_replica():
    """Round-robin pick a replica. Callers should acquire the returned lock."""
    global _rr_counter
    with _rr_lock:
        idx = _rr_counter % len(_models)
        _rr_counter += 1
    return _models[idx]


def chat_batch(messages_list, enable_thinking: bool = False,
               max_new_tokens: int = 512) -> List[str]:
    """Batched greedy generation. Uses a single replica (round-robin chosen)."""
    prompts = [
        tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for m in messages_list
    ]
    model, device, lock = _pick_replica()
    with lock:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=4096).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    gens = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gens, skip_special_tokens=True)


def extract_json(text: str) -> dict:
    cleaned = _THINK_RE.sub("", text).strip()
    m = _JSON_RE.search(cleaned)
    if not m:
        raise ValueError(f"No JSON object in model output:\n{text[:1000]}")
    return json.loads(m.group(0))


# --------------------------------------------------------------------------- #
# Prompts (unchanged)
# --------------------------------------------------------------------------- #
QUERY_SYS = (
    "You generate web search queries that verify and expand a news headline.\n"
    "Given a headline and first paragraph, produce 1-2 concise search queries that would retrieve:\n"
    "(a) authoritative confirmation of the event, and\n"
    "(b) the precise date AND time-of-day the event occurred.\n\n"
    'Output strictly JSON: {"queries": ["...", "..."]}. No other text.'
)

SYNTH_SYS = (
    "You write structured event records for a financial world model that predicts market movements.\n"
    "Given a headline, its first paragraph, the article's publish date, and web search results, output a SINGLE JSON object matching this schema:\n\n"
    "{\n"
    '  "event_id": str,\n'
    '  "headline": str,\n'
    '  "date": "YYYY-MM-DD" (article publish date),\n'
    '  "event_time": ISO-8601 datetime with timezone (e.g. "2026-04-17T14:37:00Z") | null,\n'
    '  "category": list drawn from {monetary_policy, geopolitical, financial_stress, sovereign_stress, commodity_shock, corporate_stress, policy_shock, disaster_health, other},\n'
    '  "summary": str (see SUMMARY INSTRUCTIONS below),\n'
    '  "key_entities": list of orgs/people/countries,\n'
    '  "affected_regions": list,\n'
    '  "affected_assets": list of commodities/asset classes/sectors,\n'
    '  "surprise_level": one of {expected, partially_expected, surprise, unknown},\n'
    '  "scope": one of {local, regional, global},\n'
    '  "sources": list of URLs,\n'
    '  "confidence": float in [0, 1]\n'
    "}\n\n"
    "SUMMARY INSTRUCTIONS:\n"
    "The summary is the most important field. Write 5-8 sentences covering ALL of the following layers:\n"
    "1. WHAT happened — the core event, with specific numbers, dates, or thresholds if available.\n"
    "2. WHY it matters — the causal mechanism through which this event affects markets or volatility.\n"
    "3. CONTEXT — relevant precedents, prior policy stance, ongoing trends, or diplomatic backdrop that a forecasting model needs to interpret the magnitude of the shock.\n"
    "4. FORWARD RISK — upcoming catalysts, decision dates, or escalation paths noted in the sources.\n"
    "If a layer has no supporting evidence in the snippets, skip it rather than inventing content. "
    "But do NOT skip a layer just because it takes more words — thoroughness is preferred over brevity.\n\n"
    "Hard rules:\n"
    "- `event_time` is the time the EVENT occurred, NOT the article publish time. If sources do not state it, use null. Never guess.\n"
    "- `summary` must be grounded in the provided snippets. Do not invent facts not supported by them.\n"
    "- Extract ALL relevant URLs from the search results into `sources` — do not drop any that informed your summary.\n"
    "- Output ONLY the JSON object. No preamble, no markdown fences, no thinking tags."
)


# --------------------------------------------------------------------------- #
# Batched pipeline
# --------------------------------------------------------------------------- #
def gen_queries_batch(items: List[dict]) -> List[Optional[List[str]]]:
    msgs = [
        [{"role": "system", "content": QUERY_SYS},
         {"role": "user",
          "content": f"HEADLINE: {it['headline']}\n\nFIRST_PARAGRAPH: {it['first_para']}"}]
        for it in items
    ]
    raws = chat_batch(msgs, enable_thinking=False, max_new_tokens=200)
    out: List[Optional[List[str]]] = []
    for raw in raws:
        try:
            out.append(QueryGenOutput(**extract_json(raw)).queries)
        except Exception as e:
            print(f"[gen_queries] parse error: {e}", file=sys.stderr)
            out.append(None)
    return out


def _gather_hits_for(queries: List[str]) -> List[Tuple[str, str, str]]:
    if not queries:
        return []
    hits: List[Tuple[str, str, str]] = []
    seen: set[str] = set()
    for q in queries:
        try:
            for title, body, url in search(q, 3):
                if url and url in seen:
                    continue
                if url:
                    seen.add(url)
                hits.append((title, body, url))
        except Exception as e:
            print(f"[search] error for {q!r}: {e}", file=sys.stderr, flush=True)
    return hits[:5]


def synthesize_batch(items_with_hits: List[dict]) -> List[Optional[EventRecord]]:
    msgs = []
    for it in items_with_hits:
        blocks = "\n\n".join(
            f"[{i + 1}] TITLE: {t}\nURL: {u}\nSNIPPET: {b}"
            for i, (t, b, u) in enumerate(it["hits"])
        ) or "(no search results)"
        user = (
            f"EVENT_ID: {it['event_id']}\nHEADLINE: {it['headline']}\n"
            f"ARTICLE_PUBLISH_DATE: {it['pub_date']}\nFIRST_PARAGRAPH: {it['first_para']}\n\n"
            f"SEARCH_RESULTS:\n{blocks}"
        )
        msgs.append([
            {"role": "system", "content": SYNTH_SYS},
            {"role": "user", "content": user},
        ])
    raws = chat_batch(msgs, enable_thinking=True, max_new_tokens=1024)
    out: List[Optional[EventRecord]] = []
    for raw in raws:
        try:
            out.append(EventRecord(**extract_json(raw)))
        except Exception as e:
            print(f"[synthesize] parse error: {e}", file=sys.stderr)
            out.append(None)
    return out


# --------------------------------------------------------------------------- #
# DataFrame glue
# --------------------------------------------------------------------------- #
def _compose_first_para(row) -> str:
    parts = []
    for col in ("abstract", "lead_paragraph", "snippet"):
        val = row.get(col)
        if val is None:
            continue
        s = str(val).strip()
        if s and s.lower() != "nan":
            parts.append(s)
    seen, uniq = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return "\n\n".join(uniq)


def _normalize_pub_date(val) -> str:
    if val is None:
        return ""
    try:
        return pd.Timestamp(val).strftime("%Y-%m-%d")
    except Exception:
        return str(val)[:10]


def _flush(results: list, out_path: Path) -> None:
    df = pd.DataFrame(results)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(out_path)


def _row_to_item(idx, row) -> Optional[dict]:
    headline = str(row.get("headline") or "").strip()
    if not headline:
        return None
    event_id = str(row.get("url") or f"evt-{idx}")
    return {
        "event_id": event_id,
        "headline": headline,
        "first_para": _compose_first_para(row),
        "pub_date": _normalize_pub_date(row.get("date")),
        "source_url": str(row.get("url") or ""),
        "section": str(row.get("section") or ""),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    run_start = time.time()
    print(f"[start] {datetime.now().isoformat(timespec='seconds')}", flush=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[data] reading {INPUT_PATH} ...", flush=True)
    df = pd.read_parquet(INPUT_PATH)
    print(f"[data] loaded {len(df)} raw rows (elapsed {_fmt_elapsed(time.time() - run_start)})", flush=True)
    if "has_commodity_impact" in df.columns:
        df = df[df["has_commodity_impact"] == True]  # noqa: E712
        print(f"[data] {len(df)} rows after has_commodity_impact filter", flush=True)
    if LIMIT is not None:
        df = df.head(LIMIT)
        print(f"[data] {len(df)} rows after LIMIT={LIMIT}", flush=True)
    print(f"[data] {len(df)} rows from {INPUT_PATH}", flush=True)

    already_done: set[str] = set()
    results: list[dict] = []
    resume_paths = []
    if RESUME_FROM_PATH is not None and RESUME_FROM_PATH.exists():
        resume_paths.append(RESUME_FROM_PATH)
    if RESUME and OUTPUT_PATH.exists():
        resume_paths.append(OUTPUT_PATH)
    for p in resume_paths:
        prior = pd.read_parquet(p)
        for rec in prior.to_dict(orient="records"):
            eid = str(rec["event_id"])
            if eid in already_done:
                continue
            already_done.add(eid)
            results.append(rec)
        print(f"[resume] {len(prior)} rows from {p}", flush=True)
    if resume_paths:
        print(f"[resume] {len(already_done)} unique event_ids already done", flush=True)

    items: list[dict] = []
    for idx, row in df.iterrows():
        it = _row_to_item(idx, row)
        if it is None or it["event_id"] in already_done:
            continue
        items.append(it)
    total_items = len(items)
    n_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[data] {total_items} rows to process across {n_batches} batches "
          f"(BATCH_SIZE={BATCH_SIZE}, SYNTH_BATCH_SIZE={SYNTH_BATCH_SIZE})", flush=True)

    print(f"[model] loading {MODEL_ID} ...", flush=True)
    model_t0 = time.time()
    load_model(MODEL_ID)
    print(f"[model] ready in {_fmt_elapsed(time.time() - model_t0)} "
          f"(total elapsed {_fmt_elapsed(time.time() - run_start)})", flush=True)

    failures: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=SEARCH_WORKERS)
    try:
        with tqdm(total=len(items), desc="stage2") as pbar:
            for start in range(0, len(items), BATCH_SIZE):
                batch = items[start:start + BATCH_SIZE]

                queries_list = gen_queries_batch(batch)
                for it, qs in zip(batch, queries_list):
                    it["queries"] = qs or []

                hits_futures = [
                    pool.submit(_gather_hits_for, it["queries"])
                    for it in batch
                ]
                for it, f in zip(batch, hits_futures):
                    try:
                        it["hits"] = f.result()
                    except Exception as e:
                        it["hits"] = []
                        print(f"[search] batch error for {it['event_id']}: {e}",
                              file=sys.stderr, flush=True)

                records: List[Optional[EventRecord]] = []
                for s in range(0, len(batch), SYNTH_BATCH_SIZE):
                    records.extend(synthesize_batch(batch[s:s + SYNTH_BATCH_SIZE]))
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            with torch.cuda.device(i):
                                torch.cuda.empty_cache()

                for it, rec in zip(batch, records):
                    if rec is None:
                        failures.append({
                            "event_id": it["event_id"],
                            "headline": it["headline"],
                            "error": "synthesize failed",
                        })
                        pbar.update(1)
                        continue
                    if not rec.sources:
                        rec.sources = [u for _, _, u in it["hits"] if u]
                    out_rec = rec.model_dump()
                    out_rec["queries"] = list(it["queries"])
                    out_rec["n_hits"] = len(it["hits"])
                    out_rec["source_url"] = it["source_url"]
                    out_rec["section"] = it["section"]
                    results.append(out_rec)
                    already_done.add(it["event_id"])
                    pbar.update(1)

                _flush(results, OUTPUT_PATH)
                pbar.set_postfix(
                    ok=len(results),
                    fail=len(failures),
                    wiki=_wiki_calls,
                    tavily=_tavily_calls,
                )
    except KeyboardInterrupt:
        print("\n[interrupt] flushing partial results...", file=sys.stderr, flush=True)
    finally:
        pool.shutdown(wait=False)

    _flush(results, OUTPUT_PATH)
    print(f"[done] wrote {len(results)} rows -> {OUTPUT_PATH} "
          f"(failures: {len(failures)}) | "
          f"total elapsed {_fmt_elapsed(time.time() - run_start)}", flush=True)
    if failures:
        fail_path = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_failures.json")
        fail_path.write_text(json.dumps(failures, indent=2))
        print(f"[done] failures logged -> {fail_path}", flush=True)


if __name__ == "__main__":
    main()
