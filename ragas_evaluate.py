#!/usr/bin/env python3
"""
ragas_evaluator.py
==================
Loads the golden_dataset.json, queries your running RAG API for each entry,
then evaluates the results with RAGAS.

Metrics computed
----------------
  answer_relevancy     – does the answer address the question?
  faithfulness         – is the answer grounded in the retrieved context?
  context_recall       – did retrieval surface the right information?
  context_precision    – were retrieved chunks actually used?

Usage
-----
  pip install ragas datasets langchain-openai python-dotenv requests rich

  # With OpenAI as the judge LLM (default):
  OPENAI_API_KEY=sk-... python ragas_evaluator.py

  # With a custom API base (e.g. Groq-compatible):
  OPENAI_API_KEY=... JUDGE_BASE_URL=https://api.groq.com/openai/v1 \
  JUDGE_MODEL=llama-3.3-70b-versatile python ragas_evaluator.py

  # Filter to one document:
  python ragas_evaluator.py --doc predictive_analysis.pdf

  # Filter to one query type:
  python ragas_evaluator.py --type comparison

  # Skip slow tests:
  python ragas_evaluator.py --skip-types multi_hop comparison
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (override with env vars or CLI flags)
# ─────────────────────────────────────────────────────────────────────────────
RAG_BASE_URL   = os.getenv("RAG_BASE_URL",   "http://localhost:8010")
GOLDEN_DATASET = os.getenv("GOLDEN_DATASET", "golden_dataset.json")
JUDGE_MODEL    = os.getenv("JUDGE_MODEL",    "gpt-4o-mini")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", None)   # None = default OpenAI

# Map filename → file_hash.  Populate once at startup by querying /document status
# or hard-code here if you already know the hashes.
DOC_HASH_MAP: dict[str, str] = {}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: resolve file hashes from the RAG API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_hashes(base_url: str, filenames: list[str]) -> dict[str, str]:
    """
    Try to resolve filename → file_hash by calling GET /documents (if that
    endpoint exists) or fall back to reading a local hash_map.json.
    Returns a dict {filename: file_hash}.
    """
    result = {}

    # Try GET /documents first (not in the default routes, but common extension)
    try:
        r = requests.get(f"{base_url}/documents", timeout=5)
        if r.status_code == 200:
            for doc in r.json():
                result[doc["filename"]] = doc["file_hash"]
            return result
    except Exception:
        pass

    # Fall back: read hash_map.json in CWD
    hmap_path = Path("hash_map.json")
    if hmap_path.exists():
        with open(hmap_path) as f:
            data = json.load(f)
        for entry in data:
            result[entry["filename"]] = entry["file_hash"]
        console.print(f"[dim]Loaded {len(result)} hash(es) from hash_map.json[/dim]")
        return result

    console.print(
        "[yellow]⚠  Could not auto-resolve file hashes.  "
        "Create hash_map.json: [{\"filename\": \"x.pdf\", \"file_hash\": \"abc...\"}][/yellow]"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: query the RAG API
# ─────────────────────────────────────────────────────────────────────────────

def query_rag(
    base_url: str,
    question: str,
    source_doc: str,
    hash_map: dict[str, str],
) -> tuple[str, list[str]]:
    """
    POST /chat with appropriate active_documents.
    Returns (answer, list_of_context_strings).
    Context strings are reconstructed from citations since the API
    does not expose raw retrieved chunks.
    """
    if source_doc == "all":
        active_docs = [
            {"file_hash": h, "filename": fn}
            for fn, h in hash_map.items()
        ]
    else:
        fhash = hash_map.get(source_doc)
        if not fhash:
            return f"[HASH NOT FOUND for {source_doc}]", []
        active_docs = [{"file_hash": fhash, "filename": source_doc}]

    payload = {
        "query": question,
        "thread_id": str(uuid.uuid4()),
        "active_documents": active_docs,
        "active_file_hashes": [d["file_hash"] for d in active_docs],
    }

    try:
        r = requests.post(f"{base_url}/chat", json=payload, timeout=120)
        r.raise_for_status()
        body = r.json()
        answer = body.get("answer", "")
        # Build context list from citations (page reference strings)
        citations = body.get("citations", [])
        context = [
            f"{c.get('source_file','?')} page {c.get('page_number','?')}"
            for c in citations
        ]
        return answer, context
    except Exception as e:
        return f"[RAG ERROR: {e}]", []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: build RAGAS Dataset and run metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas(samples: list[dict]) -> dict:
    """
    samples: list of {question, answer, contexts, ground_truth}
    Returns dict of metric_name → score.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
        )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Run:  pip install ragas datasets langchain-openai")
        sys.exit(1)

    # Build judge LLM — supports any OpenAI-compatible endpoint
    judge_kwargs = {"model": JUDGE_MODEL, "temperature": 0}
    if JUDGE_BASE_URL:
        judge_kwargs["base_url"] = JUDGE_BASE_URL

    llm   = LangchainLLMWrapper(ChatOpenAI(**judge_kwargs))
    emb   = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    dataset = Dataset.from_list(samples)

    result = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision],
        llm=llm,
        embeddings=emb,
        raise_exceptions=False,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def score_colour(v: Optional[float]) -> str:
    if v is None:
        return "dim"
    if v >= 0.75:
        return "green"
    if v >= 0.5:
        return "yellow"
    return "red"


def print_per_sample_table(samples: list[dict], answers: list[str]):
    t = Table(title="Per-Question Results", show_lines=True)
    t.add_column("ID",       width=10)
    t.add_column("Type",     width=14)
    t.add_column("Question", width=40, overflow="fold")
    t.add_column("Answer",   width=50, overflow="fold")

    for s, ans in zip(samples, answers):
        t.add_row(
            s.get("_id", ""),
            s.get("_query_type", ""),
            s["question"][:120],
            ans[:200] if ans else "[dim]–[/dim]",
        )
    console.print(t)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for OpenMultiRAG")
    parser.add_argument("--doc",        help="Filter to a single source_doc")
    parser.add_argument("--type",       help="Filter to a single query_type")
    parser.add_argument("--skip-types", nargs="*", default=[],
                        help="Query types to skip (e.g. comparison multi_hop)")
    parser.add_argument("--no-ragas",   action="store_true",
                        help="Only query the RAG, skip RAGAS scoring")
    parser.add_argument("--base-url",   default=RAG_BASE_URL)
    parser.add_argument("--dataset",    default=GOLDEN_DATASET)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    # Load golden dataset
    with open(args.dataset) as f:
        golden = json.load(f)
    console.print(f"[bold]Loaded {len(golden)} golden entries from {args.dataset}[/bold]")

    # Apply filters
    if args.doc:
        golden = [g for g in golden if g["source_doc"] == args.doc]
    if args.type:
        golden = [g for g in golden if g["query_type"] == args.type]
    if args.skip_types:
        golden = [g for g in golden if g["query_type"] not in args.skip_types]

    console.print(f"Running evaluation on [bold]{len(golden)}[/bold] entries\n")

    # Resolve hashes
    unique_docs = {g["source_doc"] for g in golden if g["source_doc"] != "all"}
    hash_map = fetch_all_hashes(base_url, list(unique_docs))

    # ── Query RAG ────────────────────────────────────────────────────────────
    console.rule("Querying RAG API")
    ragas_samples = []
    raw_answers   = []

    for i, entry in enumerate(golden, 1):
        console.print(f"  [{i}/{len(golden)}] {entry['id']}  {entry['query'][:70]}...")

        answer, contexts = query_rag(
            base_url,
            entry["query"],
            entry["source_doc"],
            hash_map,
        )

        raw_answers.append(answer)
        ragas_samples.append({
            "question":     entry["query"],
            "answer":       answer,
            "contexts":     contexts if contexts else ["[no context retrieved]"],
            "ground_truth": entry["ground_truth"],
            # metadata carried through for display
            "_id":          entry["id"],
            "_query_type":  entry["query_type"],
        })

    print_per_sample_table(ragas_samples, raw_answers)

    if args.no_ragas:
        console.print("\n[yellow]--no-ragas flag set, skipping RAGAS scoring.[/yellow]")
        return

    # ── RAGAS scoring ────────────────────────────────────────────────────────
    console.rule("Running RAGAS Metrics")

    # Strip metadata keys before passing to RAGAS
    clean_samples = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in ragas_samples
    ]

    result = run_ragas(clean_samples)

    # ── Print summary ─────────────────────────────────────────────────────────
    console.rule("Evaluation Summary")

    metric_map = {
        "answer_relevancy":  result.get("answer_relevancy"),
        "faithfulness":      result.get("faithfulness"),
        "context_recall":    result.get("context_recall"),
        "context_precision": result.get("context_precision"),
    }

    summary = Table(title="RAGAS Scores", show_header=True)
    summary.add_column("Metric",    width=22)
    summary.add_column("Score",     width=8)
    summary.add_column("Meaning",   width=50)

    meanings = {
        "answer_relevancy":  "Does the answer actually address the question?",
        "faithfulness":      "Is the answer grounded in retrieved context (no hallucination)?",
        "context_recall":    "Did retrieval surface all ground-truth information?",
        "context_precision": "Were retrieved chunks genuinely useful?",
    }

    for metric, score in metric_map.items():
        val  = f"{score:.3f}" if score is not None else "N/A"
        col  = score_colour(score)
        summary.add_row(
            metric,
            f"[{col}]{val}[/{col}]",
            meanings.get(metric, ""),
        )

    console.print(summary)

    # Save results
    out = {
        "scores": {k: (round(v, 4) if v else None) for k, v in metric_map.items()},
        "n_samples": len(golden),
        "filter_doc":  args.doc,
        "filter_type": args.type,
        "per_sample": [
            {
                "id":           s["_id"],
                "query_type":   s["_query_type"],
                "question":     s["question"],
                "answer":       s["answer"][:300],
                "ground_truth": s["ground_truth"][:300],
            }
            for s in ragas_samples
        ],
    }
    out_path = "ragas_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    console.print(f"\n[dim]Full results saved to {out_path}[/dim]")


if __name__ == "__main__":
    main()