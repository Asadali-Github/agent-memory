#!/usr/bin/env python3
"""Build the four Chapter-4 figures from the merged run records.

    python scripts/make_ch4_figures.py --outdir writing/latex/figures

Produces fig4_1_cost_performance, fig4_2_per_ability,
fig4_3_success_by_condition, fig4_4_failure_modes — the two-panel layouts the
Ch4 captions describe. Supersedes the ad-hoc figures used before the ordering
control arms existed.
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B_ORDER = ["b", "b_reversed", "m1", "m1_chrono", "m2", "m3", "m3_chrono", "m4", "m4_chrono"]
A_ORDER = ["b", "m1", "m2", "m3", "m4"]
LABEL = {"b": "B", "b_reversed": "B-rev", "m1": "M1", "m1_chrono": "M1-ch",
         "m2": "M2", "m3": "M3", "m3_chrono": "M3-ch", "m4": "M4", "m4_chrono": "M4-ch"}
CTRL = {"b_reversed", "m1_chrono", "m3_chrono", "m4_chrono"}
RNG = np.random.default_rng(7)


def load(path):
    d = collections.defaultdict(dict)
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            d[r["condition"]][(r["item_id"], r["seed"])] = r
    return d


def boot(recs, n=3000):
    eps = collections.defaultdict(list)
    for r in recs:
        eps[str(r["item_id"]).split(":")[0]].append(r["success"])
    keys = list(eps)
    out = [np.mean([x for i in RNG.choice(len(keys), len(keys)) for x in eps[keys[i]]])
           for _ in range(n)]
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def fig_success(B, A, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [9, 5]})
    for ax, d, order, title in ((axes[0], B, B_ORDER, "Mode B — Project-Tracker"),
                                (axes[1], A, A_ORDER, "Mode A — LongMemEval$_s$")):
        vals, los, his, cols = [], [], [], []
        for c in order:
            recs = list(d[c].values())
            m = np.mean([r["success"] for r in recs])
            lo, hi = boot(recs)
            vals.append(m); los.append(m - lo); his.append(hi - m)
            cols.append("#7f8c8d" if c == "b" else ("#2980b9" if c in CTRL else "#95a5a6"))
        x = np.arange(len(order))
        ax.bar(x, vals, yerr=[los, his], capsize=3, color=cols,
               edgecolor="black", linewidth=0.6)
        ax.axhline(vals[0], ls="--", lw=1, color="#c0392b", zorder=0)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in order])
        ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=11)
        ax.set_ylabel("Task success")
        ax.grid(axis="y", alpha=0.3, lw=0.5)
    axes[0].text(0.015, 0.975, "blue = ordering-control arms\ndashed = baseline",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=8,
                 color="#444", bbox=dict(facecolor="white", edgecolor="#ccc",
                                         boxstyle="round,pad=0.35", alpha=0.9))
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_cost(B, A, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, d, order, title, logx in ((axes[0], B, B_ORDER, "Mode B", False),
                                      (axes[1], A, A_ORDER, "Mode A (log scale)", True)):
        pts = []
        for c in order:
            recs = [r for r in d[c].values() if r.get("total_prompt_tokens")]
            if not recs:
                continue
            tot = np.mean([(r.get("total_prompt_tokens") or 0)
                           + (r.get("total_completion_tokens") or 0) for r in recs])
            s = np.mean([r["success"] for r in recs])
            col = "#7f8c8d" if c == "b" else ("#2980b9" if c in CTRL else "#c0392b")
            ax.scatter(tot, s, s=140, color=col, edgecolor="black", zorder=3)
            pts.append((tot, s, c))
        # stagger labels that would collide (several arms share a cost/accuracy point)
        placed = []
        for tot, s, c in sorted(pts, key=lambda p: (-p[1], p[0])):
            dy = 6
            while any(abs(s - ps) < 0.012 and abs(dy - pd) < 11 for ps, pd in placed):
                dy -= 12
            placed.append((s, dy))
            ax.annotate(LABEL[c], (tot, s), textcoords="offset points",
                        xytext=(9, dy), fontsize=9)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel("Total tokens per item (measured, incl. write path)")
        ax.set_ylabel("Task success"); ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_ability(A, out):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for c, items in A.items():
        for r in items.values():
            t = (r.get("attr") or {}).get("question_type") or r.get("question_type") or "unknown"
            per[t][c].append(r["success"])
    abilities = sorted(per, key=lambda t: -np.mean(per[t]["m1"] or [0]))
    x = np.arange(len(abilities)); w = 0.16
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, c in enumerate(A_ORDER):
        vals = [np.mean(per[t][c]) if per[t][c] else 0 for t in abilities]
        ax.bar(x + (i - 2) * w, vals, w, label=LABEL[c], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("single-session-", "ss-") for t in abilities],
                       rotation=20, ha="right")
    ax.set_ylabel("Task success"); ax.set_ylim(0, 1.0)
    ax.set_title("Mode A success by ability type", fontsize=11)
    ax.legend(ncol=5, fontsize=9); ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_failures(B, A, out):
    tags = ["retrieval_miss", "partial_retrieval", "stale_fact", "reasoning_failure", "no_answer"]
    colours = ["#c0392b", "#e67e22", "#f1c40f", "#8e44ad", "#7f8c8d"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [9, 5]})
    for ax, d, order, title in ((axes[0], B, B_ORDER, "Mode B"),
                                (axes[1], A, A_ORDER, "Mode A")):
        x = np.arange(len(order)); bottom = np.zeros(len(order))
        for tag, col in zip(tags, colours):
            vals = []
            for c in order:
                items = d[c]; n = len(items)
                vals.append(sum(1 for r in items.values()
                                if (r.get("failure_tag") or "") == tag) / n)
            ax.bar(x, vals, bottom=bottom, color=col, label=tag.replace("_", " "),
                   edgecolor="black", linewidth=0.4)
            bottom += np.array(vals)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in order])
        ax.set_ylabel("Fraction of items"); ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=0.3, lw=0.5)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeb", default="results/rerun_2026_07_29/custom_ds_merged.jsonl")
    ap.add_argument("--modea", default="results/rerun_2026_07_29/longmemeval_s_merged.jsonl")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    B, A = load(a.modeb), load(a.modea)
    fig_success(B, A, out / "fig4_3_success_by_condition.png")
    fig_cost(B, A, out / "fig4_1_cost_performance.png")
    fig_ability(A, out / "fig4_2_per_ability.png")
    fig_failures(B, A, out / "fig4_4_failure_modes.png")
    print(f"wrote 4 figures -> {out}")
