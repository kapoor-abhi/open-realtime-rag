#!/usr/bin/env python3
"""
Generate a minimalist system architecture diagram for OpenMultiRAG.

Output:
  screenshots/architecture_minimal.png

Design goals:
  - Clean, recruiter-friendly diagram
  - Minimal colors and visual noise
  - Legible at slide/README scale
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "screenshots" / "architecture_minimal.png"


def _box(ax, x, y, w, h, title, subtitle=None, fc="#FFFFFF", ec="#111827", lw=1.2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="semibold",
        color="#111827",
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=10.2,
            color="#374151",
        )


def _arrow(ax, x1, y1, x2, y2, color="#111827", lw=1.1, style="-|>", mutation_scale=12):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arr)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Canvas
    fig = plt.figure(figsize=(14, 8), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Palette (minimal)
    service_fc = "#EFF6FF"   # very light blue
    data_fc = "#F9FAFB"      # light gray
    external_fc = "#ECFDF5"  # very light green
    obs_fc = "#FFF7ED"       # very light orange

    # Layout coordinates (x, y are bottom-left)
    _box(ax, 0.06, 0.72, 0.22, 0.16, "Streamlit UI", "Upload • Chat • Citations", fc=service_fc)
    _box(ax, 0.35, 0.72, 0.22, 0.16, "FastAPI API", "/upload • /chat • SSE", fc=service_fc)
    _box(ax, 0.64, 0.72, 0.30, 0.16, "LangGraph Workflow", "rewrite → intent → cache → retrieve → generate", fc=service_fc)

    _box(ax, 0.35, 0.46, 0.22, 0.16, "RQ Worker", "Parse • Diff • Index", fc=service_fc)

    _box(ax, 0.06, 0.22, 0.22, 0.18, "PostgreSQL", "documents • BM25 corpus • chunk_index\nLangGraph checkpointer", fc=data_fc)
    _box(ax, 0.35, 0.22, 0.22, 0.18, "Redis", "RQ broker • pub/sub status", fc=data_fc)
    _box(ax, 0.64, 0.22, 0.15, 0.18, "Qdrant", "Vectors • Semantic cache", fc=data_fc)
    _box(ax, 0.81, 0.22, 0.13, 0.18, "MinIO", "Images (S3)", fc=data_fc)

    _box(ax, 0.64, 0.46, 0.15, 0.16, "Cohere", "Embeddings • Rerank", fc=external_fc, ec="#065F46")
    _box(ax, 0.81, 0.46, 0.13, 0.16, "Groq", "LLMs • Vision", fc=external_fc, ec="#065F46")

    _box(ax, 0.06, 0.46, 0.22, 0.16, "Langfuse", "Tracing • Token/latency\n(Flush in worker)", fc=obs_fc, ec="#9A3412")

    # Arrows (main flows)
    _arrow(ax, 0.28, 0.80, 0.35, 0.80)  # UI -> API
    _arrow(ax, 0.57, 0.80, 0.64, 0.80)  # API -> LangGraph

    _arrow(ax, 0.46, 0.72, 0.46, 0.62)  # API -> Worker enqueue (conceptual)
    _arrow(ax, 0.46, 0.62, 0.46, 0.55)  # down into worker

    # API dependencies
    _arrow(ax, 0.35, 0.76, 0.28, 0.31, color="#374151")  # API -> Postgres
    _arrow(ax, 0.35, 0.72, 0.46, 0.40, color="#374151")  # API -> Redis (SSE/pubs)
    _arrow(ax, 0.64, 0.72, 0.72, 0.40, color="#374151")  # LangGraph -> Qdrant
    _arrow(ax, 0.64, 0.72, 0.715, 0.54, color="#065F46")  # LangGraph -> Cohere
    _arrow(ax, 0.64, 0.72, 0.875, 0.54, color="#065F46")  # LangGraph -> Groq

    # Worker dependencies
    _arrow(ax, 0.46, 0.46, 0.28, 0.31, color="#374151")  # Worker -> Postgres
    _arrow(ax, 0.46, 0.46, 0.46, 0.40, color="#374151")  # Worker -> Redis
    _arrow(ax, 0.57, 0.54, 0.72, 0.40, color="#374151")  # Worker -> Qdrant
    _arrow(ax, 0.57, 0.54, 0.875, 0.54, color="#065F46")  # Worker -> Groq Vision
    _arrow(ax, 0.57, 0.54, 0.875, 0.31, color="#374151")  # Worker -> MinIO

    # Observability
    _arrow(ax, 0.35, 0.76, 0.17, 0.54, color="#9A3412", style="->", mutation_scale=10)
    _arrow(ax, 0.35, 0.54, 0.28, 0.54, color="#9A3412", style="->", mutation_scale=10)

    ax.text(
        0.06,
        0.94,
        "OpenMultiRAG — Realtime Multimodal RAG (System Architecture)",
        fontsize=16,
        fontweight="semibold",
        color="#111827",
        ha="left",
        va="center",
    )
    ax.text(
        0.06,
        0.91,
        "Minimalist diagram: asynchronous ingestion, hybrid retrieval, scope-aware cache, and observability.",
        fontsize=10.5,
        color="#4B5563",
        ha="left",
        va="center",
    )

    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()

