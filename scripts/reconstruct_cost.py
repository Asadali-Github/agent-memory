#!/usr/bin/env python
"""Reconstruct TRUE per-item LLM cost, including the write path (B3).

    python scripts/reconstruct_cost.py --file results/runs/custom_ds.jsonl

WHY THIS EXISTS
Ch3 defines token cost as "total prompt and completion tokens per item", but
`prompt_tokens` in the run records is the ANSWERING call only. M3 additionally
makes ~21.6 extraction calls per item, none of which are counted. The reported
"45% token saving" is therefore a saving on the answering call, not on cost as
Ch3 defines it, and the Ch5 "strictly better default" / efficient-frontier
argument rests on it.

A NOTE ON memory_written_tokens
The expert review treated M3's mean `memory_written_tokens` (~1,171) as its
write-path cost. It is not: every condition logs the same figure — the baseline
logs ~1,171 too — because it counts tokens written into the LOCAL store, which
for B/M1/M4 is a Python list and costs nothing. Adding it to M3 alone would
overstate the correction. The real uncounted spend is the extraction CALLS.

WHAT THIS SCRIPT DOES
Reconstructs extraction spend analytically from what IS logged:
    extraction prompt  ~= template + (extract_every turns x mean tokens/turn)
    extraction output  ~= facts emitted, bounded by the per-call cap
    per item           = overhead_calls x (prompt + output)
This is an ESTIMATE with stated assumptions and it is the acceptable route only
if you say so in Ch3. The clean route is to re-run M3 with the P3-metered runner
(`scripts/run_remediation.py --with-m3`), which logs total_prompt_tokens
directly. If those columns are present this script reports the measured figures
and flags the estimate as unnecessary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# EXTRACT_PROMPT in semantic.py, minus the {chunk} placeholder.
EXTRACT_TEMPLATE_TOKENS = 45
# Each excerpt line is prefixed "[role step N] ".
LINE_PREFIX_TOKENS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results/runs/custom_ds.jsonl")
    ap.add_argument("--extract-every", type=int, default=5,
                    help="must match configs/conditions/m3.yaml")
    ap.add_argument("--completion-per-call", type=int, default=60,
                    help="assumed mean extractor output tokens per call")
    args = ap.parse_args()

    path = Path(args.file)
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    df = pd.DataFrame(rows)

    if "total_prompt_tokens" in df.columns and df["total_prompt_tokens"].notna().any():
        print("total_prompt_tokens is present — reporting MEASURED cost.\n")
        g = df.groupby("condition").agg(
            measured_prompt=("total_prompt_tokens", "mean"),
            measured_completion=("total_completion_tokens", "mean"),
            llm_calls=("total_llm_calls", "mean")).round(1)
        g["total"] = (g.measured_prompt + g.measured_completion).round(1)
        base = g.loc["b", "total"]
        g["vs_baseline_%"] = (100 * (g.total - base) / base).round(1)
        print(g.to_string())
        return

    print("total_prompt_tokens NOT logged — reconstructing analytically.")
    print(f"assumptions: extract_every={args.extract_every}, "
          f"template={EXTRACT_TEMPLATE_TOKENS} tok/call, "
          f"output={args.completion_per_call} tok/call\n")

    df["tokens_per_turn"] = df["memory_written_tokens"] / df["memory_items"].clip(lower=1)

    # `overhead_calls` is memory.stats().overhead_calls — a CUMULATIVE per-episode
    # counter, so within an episode it rises 16, 17, 18, ... across queries. Using
    # it as a per-item figure charges every query for the whole episode's
    # extraction and overstates M3's spend roughly fourfold. Difference it within
    # (condition, seed, episode) ordered by query step to get the calls actually
    # incurred since the previous query — the amortisation the runner's own
    # comment describes.
    df["episode_id"] = df["item_id"].str.split(":").str[0]
    df["q_step"] = (df["item_id"].str.split(":q").str[-1]
                    .astype(float, errors="ignore"))
    df = df.sort_values(["condition", "seed", "episode_id", "q_step"])
    grp = df.groupby(["condition", "seed", "episode_id"])["overhead_calls"]
    df["overhead_calls_incremental"] = (
        grp.diff().fillna(df["overhead_calls"]).clip(lower=0))

    g = df.groupby("condition").agg(
        answer_prompt=("prompt_tokens", "mean"),
        answer_completion=("completion_tokens", "mean"),
        overhead_cumulative=("overhead_calls", "mean"),
        overhead_calls=("overhead_calls_incremental", "mean"),
        tokens_per_turn=("tokens_per_turn", "mean"),
    )

    chunk = args.extract_every * (g.tokens_per_turn + LINE_PREFIX_TOKENS)
    per_call = EXTRACT_TEMPLATE_TOKENS + chunk + args.completion_per_call
    g["extraction_est"] = (g.overhead_calls * per_call).round(1)

    g["reported_total"] = (g.answer_prompt + g.answer_completion).round(1)
    g["true_total_est"] = (g.reported_total + g.extraction_est).round(1)

    base_rep = g.loc["b", "reported_total"]
    base_true = g.loc["b", "true_total_est"]
    g["reported_vs_b_%"] = (100 * (g.reported_total - base_rep) / base_rep).round(1)
    g["TRUE_vs_b_%"] = (100 * (g.true_total_est - base_true) / base_true).round(1)

    cols = ["answer_prompt", "answer_completion", "overhead_cumulative",
            "overhead_calls", "extraction_est", "reported_total", "true_total_est",
            "reported_vs_b_%", "TRUE_vs_b_%"]
    print(g[cols].round(1).to_string())
    print("\noverhead_cumulative = the raw logged counter (whole episode to date)")
    print("overhead_calls      = incremental calls charged to THIS query (amortised)")

    if "m3" in g.index:
        rep, true = g.loc["m3", "reported_vs_b_%"], g.loc["m3", "TRUE_vs_b_%"]
        print(f"\nM3: reported {rep:+.1f}% vs baseline on the answering call alone;")
        print(f"    {true:+.1f}% once extraction calls are included.")
        if true > -10:
            print("\n    The 45% saving does not survive. Withdraw the abstract claim,")
            print("    the Ch4 'dominating the baseline outright' line, and the Ch5")
            print("    efficient-frontier / £0.0099 -> £0.0055 costing.")
        else:
            print("\n    A saving survives, but it is smaller than reported. State it")
            print("    as amortised per episode and show the write path explicitly.")

    print("\nSensitivity — TRUE_vs_b_% for M3 under other output assumptions:")
    for out_tok in (20, 40, 60, 80, 120):
        pc = EXTRACT_TEMPLATE_TOKENS + chunk + out_tok
        tot = g.reported_total + g.overhead_calls * pc
        v = 100 * (tot.loc["m3"] - tot.loc["b"]) / tot.loc["b"]
        print(f"  {out_tok:>4} tok/call -> {v:+6.1f}%")


if __name__ == "__main__":
    main()
