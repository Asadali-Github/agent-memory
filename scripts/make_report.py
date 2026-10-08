#!/usr/bin/env python
"""Full statistical battery over results/runs/*.jsonl -> results/report.md.

Per task: headline summary with Wilson CIs, primary success contrasts by EXACT
McNEMAR vs baseline (Holm-corrected), bootstrap CIs, per-seed variance, cost
contrasts on tokens and latency, logistic regression as odds ratios, and the
failure-tag crosstab. Numbers feed Ch4 directly; the report is regenerated,
never hand-edited.

SAMPLE DEFINITION. Mode B is reported at FULL n (every generated item) as the
primary analysis, with the evidence_distance >= GAP_FILTER subset alongside as a
labelled sensitivity analysis. The filter is applied in analysis, not enforced
by the generator, so it cannot be described as a design parameter; presenting
the filtered subset as primary is what made Ch4's "every value below corresponds
to the registered analysis" false.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agentmem.logging.trajectory import load_records  # noqa: E402
from agentmem.utils.manifest import git_commit  # noqa: E402
from analysis.stats import (  # noqa: E402
    cost_contrasts,
    failure_crosstab,
    logistic_regression,
    odds_ratio_table,
    per_seed_table,
    records_to_df,
    success_contrasts,
    summary_table,
)

CONDITIONS = ["b", "m1", "m2", "m3", "m4", "b_reversed"]

# Post-hoc gap filter, reported as SENSITIVITY only. Ch3 must not describe this
# as generator-enforced: the true minimum evidence_distance in the full run is 1.
GAP_FILTER = 20


def block(text: str) -> str:
    return f"```\n{text}\n```\n"


def report_task(df, task: str) -> str:
    out = [f"## Task: {task}\n"]
    out.append(f"Records: {len(df)} | seeds: {sorted(df.seed.unique().tolist())} "
               f"| conditions: {sorted(df.condition.unique().tolist())}\n")
    out.append("### Headline summary\n" + block(summary_table(df).to_string()))

    present = [c for c in CONDITIONS if c in set(df.condition)]
    others = [c for c in present if c != "b"]
    cluster = "episode_id" if "episode_id" in df.columns else None

    if others:
        out.append("### PRIMARY — success vs baseline: exact McNemar "
                   "(Holm-corrected) + risk difference with bootstrap CI\n")
        out.append("_McNemar, not Wilcoxon: on 0/1 outcomes every non-zero "
                   "difference has |d|=1 and the signed-rank statistic "
                   "degenerates._\n")
        sc = success_contrasts(df, "b", others, cluster=cluster)
        cols = ["contrast", "n_items", "n_discordant", "risk_diff", "ci_low",
                "ci_high", "odds_ratio", "p", "p_holm", "significant_05"]
        out.append(block(sc[cols].round(4).to_string(index=False)))
        out.append("_n_items is the MATCHED n. A paired test may only ever be "
                   "quoted against this number, never against the raw per-"
                   "condition record counts._\n")

        out.append("### Per-seed variance\n")
        ps = per_seed_table(df)
        out.append(block(ps.to_string()))
        risky = [f"{r['contrast']} (|effect|={abs(r['risk_diff']):.3f})"
                 for _, r in sc.iterrows()
                 if r["contrast"].split(" vs ")[0] in ps.index
                 and abs(r["risk_diff"]) < ps.loc[r["contrast"].split(" vs ")[0], "spread"]]
        if risky:
            out.append("**Effects smaller than their own seed-to-seed spread — "
                       "do not report these as findings without more seeds:** "
                       + "; ".join(risky) + "\n")

        out.append("### RQ3 — COST contrasts (tokens, latency), Holm-corrected "
                   "within metric\n")
        cc = cost_contrasts(df, "b", others)
        if len(cc):
            ccols = ["contrast", "metric", "mean_diff", "pct_change", "ci_low",
                     "ci_high", "p_holm", "significant_05"]
            out.append(block(cc[ccols].round(4).to_string(index=False)))
        if "total_prompt_tokens" not in df.columns:
            out.append("_WARNING: `prompt_tokens` counts the ANSWERING call only "
                       "and excludes extraction/write calls, so it understates "
                       "any condition with a write path. Re-run with the metered "
                       "runner, or use `scripts/reconstruct_cost.py`, before "
                       "quoting any token saving._\n")

    try:
        extra = ["question_type"] if "question_type" in df and df["question_type"].notna().any() \
            else (["evidence_distance"] if "evidence_distance" in df else [])
        model = logistic_regression(df.dropna(subset=["success"]),
                                    extra_terms=[f"C({extra[0]})"] if extra and extra[0] == "question_type"
                                    else extra,
                                    cluster=cluster)
        out.append("### Logistic regression (success ~ condition"
                   + (f" + {extra[0]}" if extra else "")
                   + (", cluster-robust SEs" if cluster else "") + ")\n"
                   + block(str(model.summary2())))
        out.append("#### Odds ratios with 95% CIs\n"
                   + block(odds_ratio_table(model).to_string()))
    except Exception as e:  # small pilots can be separable — report, don't crash
        out.append(f"### Logistic regression\n_Not estimable on this data: {e}_\n")

    out.append("### Failure-mode composition (row-normalised)\n"
               + block(failure_crosstab(df).to_string()))
    return "\n".join(out)


def main() -> None:
    parts = [
        "# Experiment report (auto-generated — do not hand-edit)\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()} | "
        f"commit: {git_commit()}\n",
    ]
    n_tasks = 0
    for f in sorted((ROOT / "results" / "runs").glob("*.jsonl")):
        df = records_to_df(load_records(f))
        if df.empty:
            continue
        # Mode B item_ids are "<episode>:q<step>"; queries within an episode
        # share a history and are NOT independent, so tests and CIs cluster here.
        if "item_id" in df.columns:
            df["episode_id"] = df["item_id"].astype(str).str.split(":").str[0]

        parts.append(report_task(df, f.stem))
        n_tasks += 1

        # Sensitivity: the post-hoc gap filter, clearly labelled as such.
        if "evidence_distance" in df.columns and df["evidence_distance"].notna().any():
            sub = df[df["evidence_distance"] >= GAP_FILTER]
            n_items_full = df["item_id"].nunique()
            n_items_sub = sub["item_id"].nunique()
            if 0 < n_items_sub < n_items_full:
                parts.append(
                    f"\n---\n\n# SENSITIVITY ANALYSIS — {f.stem}, "
                    f"evidence_distance >= {GAP_FILTER}\n\n"
                    f"Post-hoc filter applied in ANALYSIS, not enforced by the "
                    f"generator (true minimum distance in the full run is "
                    f"{int(df['evidence_distance'].min())}). Primary results are "
                    f"the full-sample section above ({n_items_full} items); this "
                    f"subset ({n_items_sub} items, median distance "
                    f"{sub['evidence_distance'].median():.0f}) is reported to show "
                    f"the conclusions' robustness to the gap threshold. Effect "
                    f"sizes move materially between the two, so neither may be "
                    f"quoted without stating which sample it came from.\n")
                parts.append(report_task(sub, f"{f.stem} (distance >= {GAP_FILTER})"))
    if not n_tasks:
        sys.exit("No run records found — run scripts/run_full_grid.py first.")
    out = ROOT / "results" / "report.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out} ({n_tasks} task section(s))")


if __name__ == "__main__":
    main()
