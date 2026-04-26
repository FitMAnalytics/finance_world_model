"""Stage 2 event expansion: filtered-headline parquet -> structured event records.

Usage (from project root, inside the venv):
    python stage2_event_expansion.py \
        --input  data/nyt_filtered/nyt_filtered_2015_01.parquet \
        --output data/nyt_stage2/nyt_stage2_2015_01.parquet

Optional:
    --limit N              process only the first N commodity-relevant rows
    --checkpoint-every N   flush partial parquet every N successful rows (default 50)
    --resume               skip rows whose event_id already exists in --output
    --model HF_ID          override model id (default: Qwen/Qwen3-4B-Instruct-2507)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote

warnings.filterwarnings("ignore", message=".*torch._check_is_size.*")

import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# --------------------------------------------------------------------------- #
# Schema
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
# Search (Wikipedia first, Google CSE fallback)
# --------------------------------------------------------------------------- #
_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_TAVILY_URL = "https://api.tavily.com/search"
_UA = {"User-Agent": "finance-world-model/0.1 (research; contact via repo)"}


def _wiki_search(query: str, k: int) -> List[Tuple[str, str, str]]:
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
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
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
# Model + chat helper
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

tokenizer = None  # type: ignore[assignment]
model = None      # type: ignore[assignment]


def load_model(model_id: str) -> None:
    global tokenizer, model
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        attn_implementation="sdpa",
        dtype=torch.float16,
    )
    model.eval()
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / 1024**3
        first = next(model.parameters()).device
        print(f"[model] {model_id} loaded: {alloc_gb:.2f} GB on {first}")
    else:
        print(f"[model] {model_id} loaded on CPU (no CUDA)")


def chat(messages, enable_thinking: bool = False, max_new_tokens: int = 512) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def extract_json(text: str) -> dict:
    cleaned = _THINK_RE.sub("", text).strip()
    m = _JSON_RE.search(cleaned)
    if not m:
        raise ValueError(f"No JSON object in model output:\n{text[:1000]}")
    return json.loads(m.group(0))


# --------------------------------------------------------------------------- #
# Pipeline
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


def gen_queries(headline: str, first_para: str) -> List[str]:
    raw = chat(
        [{"role": "system", "content": QUERY_SYS},
         {"role": "user", "content": f"HEADLINE: {headline}\n\nFIRST_PARAGRAPH: {first_para}"}],
        enable_thinking=False, max_new_tokens=200,
    )
    return QueryGenOutput(**extract_json(raw)).queries


def synthesize(event_id, headline, first_para, pub_date, hits) -> EventRecord:
    blocks = "\n\n".join(
        f"[{i + 1}] TITLE: {t}\nURL: {u}\nSNIPPET: {b}"
        for i, (t, b, u) in enumerate(hits)
    ) or "(no search results)"
    user = (
        f"EVENT_ID: {event_id}\nHEADLINE: {headline}\n"
        f"ARTICLE_PUBLISH_DATE: {pub_date}\nFIRST_PARAGRAPH: {first_para}\n\n"
        f"SEARCH_RESULTS:\n{blocks}"
    )
    raw = chat(
        [{"role": "system", "content": SYNTH_SYS},
         {"role": "user", "content": user}],
        enable_thinking=True, max_new_tokens=1024,
    )
    return EventRecord(**extract_json(raw))


def run_event(event_id, headline, first_para, pub_date):
    queries = gen_queries(headline, first_para)
    hits: List[Tuple[str, str, str]] = []
    seen = set()
    for q in queries:
        for title, body, url in search(q, k=3):
            if url and url in seen:
                continue
            seen.add(url)
            hits.append((title, body, url))
    hits = hits[:5]
    record = synthesize(event_id, headline, first_para, pub_date, hits)
    if not record.sources:
        record.sources = [u for _, _, u in hits if u]
    return record, queries, hits


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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", "-i", required=True, type=Path)
    ap.add_argument("--output", "-o", required=True, type=Path)
    ap.add_argument("--limit", "-n", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true",
                    help="skip event_ids already present in --output")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = ap.parse_args()

    load_dotenv()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    if "has_commodity_impact" in df.columns:
        df = df[df["has_commodity_impact"] == True]  # noqa: E712
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"[data] {len(df)} rows from {args.input}")

    already_done: set[str] = set()
    results: list[dict] = []
    if args.resume and args.output.exists():
        prior = pd.read_parquet(args.output)
        already_done = set(prior["event_id"].astype(str))
        results = prior.to_dict(orient="records")
        print(f"[resume] {len(already_done)} rows already in {args.output}")

    load_model(args.model)

    failures: list[dict] = []
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="stage2"):
            headline = str(row.get("headline") or "").strip()
            if not headline:
                continue
            event_id = str(row.get("url") or f"evt-{idx}")
            if event_id in already_done:
                continue
            first_para = _compose_first_para(row)
            pub_date = _normalize_pub_date(row.get("date"))

            try:
                record, queries, hits = run_event(event_id, headline, first_para, pub_date)
                rec = record.model_dump()
                rec["queries"] = list(queries)
                rec["n_hits"] = len(hits)
                rec["source_url"] = str(row.get("url") or "")
                rec["section"] = str(row.get("section") or "")
                results.append(rec)
                already_done.add(event_id)
            except Exception as e:
                failures.append({"event_id": event_id, "headline": headline, "error": repr(e)})
                continue

            if len(results) % args.checkpoint_every == 0:
                _flush(results, args.output)
    except KeyboardInterrupt:
        print("\n[interrupt] flushing partial results...", file=sys.stderr)

    _flush(results, args.output)
    print(f"[done] wrote {len(results)} rows -> {args.output} "
          f"(failures: {len(failures)})")
    if failures:
        fail_path = args.output.with_name(args.output.stem + "_failures.json")
        fail_path.write_text(json.dumps(failures, indent=2))
        print(f"[done] failures logged -> {fail_path}")


if __name__ == "__main__":
    main()
