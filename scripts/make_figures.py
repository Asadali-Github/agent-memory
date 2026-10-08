#!/usr/bin/env python
"""Rebuild every dissertation figure from results/runs/*.jsonl in one command.

  python scripts/make_figures.py            # -> results/figures/*.png

Figures (one per RQ angle):
  fig1_success_by_condition   success rate + bootstrap 95% CI, per task
  fig2_success_vs_distance    Mode B: success vs evidence distance (memory decay in practice)
  fig3_cost_performance       tokens-per-question vs success (the efficiency frontier)
  fig4_failure_modes          stacked failure-tag composition per condition
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agentmem.logging.trajectory import load_records  # noqa: E402
from analysis.stats import records_to_df  # noqa: E402

FIG_DIR = ROOT / "results" / "figures"
ORDER = ["b", "m1", "m2", "m3", "m4"]
LABELS = {"b": "B (none)", "m1": "M1 episodic", "m2": "M2 summary",
          "m3": "M3 semantic", "m4": "M4 decay"}


def _boot_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 1) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fig_success_by_condition(df: pd.DataFrame, task: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs, means, err_lo, err_hi = [], [], [], []
    for c in ORDER:
        x = df.loc[df.condition == c, "success"].to_numpy()
        if not len(x):
            continue
        lo, hi = _boot_ci(x)
        xs.append(LABELS[c]); means.append(x.mean())
        err_lo.append(x.mean() - lo); err_hi.append(hi - x.mean())
    ax.bar(xs, means, yerr=[err_lo, err_hi], capsize=4, color="#4878a8")
    ax.set_ylabel("Task success rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"Success by memory architecture - {task} (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig1_success_by_condition_{task}.png", dpi=200)
    plt.close(fig)


def fig_success_vs_distance(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["evidence_distance"]) if "evidence_distance" in df else pd.DataFrame()
    if d.empty:
        return
    bins = [0, 20, 40, 60, 80, 200]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c in ORDER:
        sub = d[d.condition == c]
        if sub.empty:
            continue
        grp = sub.groupby(pd.cut(sub["evidence_distance"], bins, right=False),
                          observed=True)["success"].mean()
        ax.plot([iv.mid for iv in grp.index], grp.values, marker="o", label=LABELS[c])
    ax.set_xlabel("Steps between evidence and question (evidence distance)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Does memory survive distance? (Mode B custom task)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_success_vs_distance.png", dpi=200)
    plt.close(fig)


def fig_cost_performance(df: pd.DataFrame, task: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c in ORDER:
        sub = df[df.condition == c]
        if sub.empty:
            continue
        cost = (sub["prompt_tokens"] + sub["completion_tokens"]).mean()
        ax.scatter(cost, sub["success"].mean(), s=90)
        ax.annotate(LABELS[c], (cost, sub["success"].mean()),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean tokens per question (prompt + completion)")
    ax.set_ylabel("Task success rate")
    ax.set_title(f"Cost-performance frontier - {task}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig3_cost_performance_{task}.png", dpi=200)
    plt.close(fig)


def fig_failure_modes(df: pd.DataFrame, task: str) -> None:
    ct = pd.crosstab(df["condition"], df["failure_tag"], normalize="index")
    ct = ct.reindex([c for c in ORDER if c in ct.index])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottom = np.zeros(len(ct))
    for tag in ct.columns:
        ax.bar([LABELS[c] for c in ct.index], ct[tag].values, bottom=bottom, label=tag)
        bottom += ct[tag].values
    ax.set_ylabel("Share of questions")
    ax.set_title(f"Failure-mode composition - {task}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig4_failure_modes_{task}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = 0
    for f in sorted((ROOT / "results" / "runs").glob("*.jsonl")):
        df = records_to_df(load_records(f))
        if df.empty:
            continue
        task = f.stem
        fig_success_by_condition(df, task)
        fig_cost_performance(df, task)
        fig_failure_modes(df, task)
        fig_success_vs_distance(df)
        made += 1
    print(f"Figures rebuilt for {made} task file(s) -> {FIG_DIR}")


if __name__ == "__main__":
    main()
