"""Statistical analysis over JSONL run records.

Protocol:
- Primary (binary outcomes): pairwise condition contrasts on task_success via
  EXACT McNEMAR over per-item paired outcomes, with Holm correction across the
  declared contrast family. Wilcoxon signed-rank is retained only for
  continuous metrics — on 0/1 data every non-zero difference has |d| = 1, the
  ranks are fully tied, and the signed-rank statistic degenerates. McNemar is
  the correct paired test for dichotomous outcomes and conditions only on the
  discordant pairs.
- Effect size + uncertainty: risk difference with a Wilson-style bootstrap CI,
  plus the paired odds ratio (b/c on the discordant cells) for binary outcomes.
- Cost (RQ3): the SAME machinery applied to token and latency columns, so cost
  is reported with the statistical care Ch2 §2.5 promises rather than as bare
  means. Continuous metrics use Wilcoxon signed-rank, which is appropriate
  there, plus a paired bootstrap CI on the mean difference.
- Secondary: logistic regression success ~ condition (+ question_type in
  Mode A, + evidence_distance in Mode B), reported as odds ratios with CIs.
- RQ4: failure-tag crosstab per condition.

Every p-value that enters the dissertation must pass through holm_correct()
with its family declared — see DECLARED_FAMILIES below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

# Cost metrics. total_prompt_tokens/total_completion_tokens are the P3-metered
# columns that include the write path (extraction calls, memory writes);
# prompt_tokens counts only the answering call and UNDERSTATES M2/M3, which is
# what produced the withdrawn "45% saving" claim. Prefer the total_* columns and
# fall back only when re-runs have not yet produced them.
COST_METRICS = ("total_prompt_tokens", "total_completion_tokens", "latency_ms")
COST_METRICS_LEGACY = ("prompt_tokens", "completion_tokens", "latency_ms")


def records_to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if not df.empty:
        df["success"] = df["success"].astype(int)
    return df


def paired_item_means(df: pd.DataFrame, cond_a: str, cond_b: str,
                      metric: str = "success") -> pd.DataFrame:
    """Per-item means (across seeds) of `metric` for two conditions, inner-joined.

    The inner join is what makes the test PAIRED. Reporting an unpaired contrast
    as paired is exactly the B4 error (Table 4.4 paired a 341-record baseline
    against a 120-record M1); `n_items` returned here is the true matched n and
    is the only n that may be quoted alongside a paired test.
    """
    piv = (df[df.condition.isin([cond_a, cond_b])]
           .groupby(["item_id", "condition"])[metric].mean().unstack())
    return piv.dropna(subset=[cond_a, cond_b])


def mcnemar_contrast(df: pd.DataFrame, cond_a: str, cond_b: str,
                     metric: str = "success") -> dict:
    """EXACT McNemar on paired binary outcomes — the primary success test.

    Items are dichotomised at 0.5 when averaged over seeds (a tie contributes to
    neither discordant cell). The exact binomial form is used rather than the
    chi-square approximation because the discordant counts here are small.

    Returns b/c (the discordant cells), the paired odds ratio b/c, and the risk
    difference. Note the p-value tests the discordant pairs only; the risk
    difference is the effect size and should always be reported with it.
    """
    piv = paired_item_means(df, cond_a, cond_b, metric)
    a = (piv[cond_a].to_numpy() > 0.5).astype(int)
    b_ = (piv[cond_b].to_numpy() > 0.5).astype(int)
    n01 = int(((a == 0) & (b_ == 1)).sum())   # only cond_b succeeds
    n10 = int(((a == 1) & (b_ == 0)).sum())   # only cond_a succeeds
    n_disc = n01 + n10
    p = 1.0 if n_disc == 0 else float(
        sps.binomtest(n10, n_disc, 0.5, alternative="two-sided").pvalue)
    odds = float("inf") if n01 == 0 and n10 > 0 else (
        0.0 if n10 == 0 and n01 > 0 else (float(n10 / n01) if n01 else float("nan")))
    return {"contrast": f"{cond_a} vs {cond_b}", "test": "exact McNemar",
            "n_items": len(piv), "n_discordant": n_disc,
            "n_only_a": n10, "n_only_b": n01,
            "risk_diff": float(piv[cond_a].mean() - piv[cond_b].mean()),
            "odds_ratio": odds, "p": p}


def wilcoxon_contrast(df: pd.DataFrame, cond_a: str, cond_b: str,
                      metric: str = "success") -> dict:
    """Paired Wilcoxon signed-rank on any metric column.

    CORRECT for continuous metrics (tokens, latency). For binary success use
    mcnemar_contrast(): on 0/1 data all non-zero |d| equal 1, ranks tie
    completely, and the signed-rank statistic degenerates.
    """
    piv = paired_item_means(df, cond_a, cond_b, metric)
    a, b = piv[cond_a].to_numpy(), piv[cond_b].to_numpy()
    diff = a - b
    base = {"contrast": f"{cond_a} vs {cond_b}", "metric": metric,
            "test": "Wilcoxon signed-rank", "n_items": len(piv)}
    if len(piv) == 0 or np.allclose(diff, 0):
        return {**base, "mean_diff": 0.0, "median_diff": 0.0, "W": None, "p": 1.0}
    W, p = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {**base, "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)), "W": float(W), "p": float(p)}


def bootstrap_diff_ci(
    df: pd.DataFrame, cond_a: str, cond_b: str, metric: str = "success",
    n_boot: int = 10_000, seed: int = 1, alpha: float = 0.05,
    cluster: str | None = None,
) -> dict:
    """Paired bootstrap CI on the mean difference in `metric`.

    `cluster` resamples whole clusters (e.g. "episode_id") instead of items.
    Item-level resampling treats queries from the same episode as independent
    when they are not, which narrows the interval — the same pseudo-replication
    that inflated the Mode A CI in B4. Pass cluster="episode_id" for Mode B.
    """
    piv = paired_item_means(df, cond_a, cond_b, metric)
    diff = (piv[cond_a] - piv[cond_b])
    rng = np.random.default_rng(seed)

    if cluster and cluster in df.columns:
        key = (df.drop_duplicates("item_id").set_index("item_id")[cluster]
               .reindex(diff.index))
        groups = [g.to_numpy() for _, g in diff.groupby(key)]
        k = len(groups)
        boots = np.array([
            np.concatenate([groups[i] for i in rng.integers(0, k, k)]).mean()
            for _ in range(n_boot)])
    else:
        d = diff.to_numpy()
        n = len(d)
        boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])

    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"contrast": f"{cond_a} vs {cond_b}", "metric": metric,
            "mean_diff": float(diff.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "n_items": len(diff),
            "resampled": cluster or "item"}


def cost_contrasts(df: pd.DataFrame, baseline: str = "b",
                   conditions: list[str] | None = None,
                   metrics: tuple[str, ...] | None = None) -> pd.DataFrame:
    """RQ3: paired tests + CIs on cost metrics, Holm-corrected WITHIN metric.

    This is what Ch2 §2.5 promises ("cost reported with the same statistical
    care as task success") and what Table 4.1 currently does not deliver.

    Holm families are per-metric: correcting tokens and latency jointly would
    be over-conservative, since they are separate pre-declared questions.
    """
    if metrics is None:
        # Pick the metered set only if the P3 total_* columns are actually
        # present. Testing membership metric-by-metric would silently select
        # just latency_ms (it appears in both tuples) and drop the token tests.
        metered = [m for m in COST_METRICS if m in df.columns]
        use_metered = "total_prompt_tokens" in df.columns
        metrics = tuple(metered) if use_metered else \
            tuple(m for m in COST_METRICS_LEGACY if m in df.columns)
    conds = conditions or [c for c in sorted(df.condition.unique()) if c != baseline]

    rows: list[dict] = []
    for metric in metrics:
        if metric not in df.columns or df[metric].isna().all():
            continue
        fam = [wilcoxon_contrast(df, c, baseline, metric) for c in conds]
        fam = holm_correct(fam)
        for r, c in zip(fam, conds):
            ci = bootstrap_diff_ci(df, c, baseline, metric)
            base_mean = df[df.condition == baseline][metric].mean()
            rows.append({**r, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                         "pct_change": (100.0 * r["mean_diff"] / base_mean
                                        if base_mean else float("nan"))})
    return pd.DataFrame(rows)


def success_contrasts(df: pd.DataFrame, baseline: str = "b",
                      conditions: list[str] | None = None,
                      cluster: str | None = None) -> pd.DataFrame:
    """RQ1/RQ2: the primary success family — exact McNemar + risk difference + CI.

    One declared Holm family covering every condition-vs-baseline contrast in
    this mode. Use this instead of assembling contrasts by hand, so nothing
    escapes correction (the previous family covered only the four Mode B
    contrasts and left Mode A and the logit coefficients uncorrected).
    """
    conds = conditions or [c for c in sorted(df.condition.unique()) if c != baseline]
    fam = holm_correct([mcnemar_contrast(df, c, baseline) for c in conds])
    rows = []
    for r, c in zip(fam, conds):
        ci = bootstrap_diff_ci(df, c, baseline, "success", cluster=cluster)
        rows.append({**r, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                     "resampled": ci["resampled"]})
    return pd.DataFrame(rows)


def per_seed_table(df: pd.DataFrame, metric: str = "success") -> pd.DataFrame:
    """Per-seed means per condition + the spread across seeds (A5).

    Seed spread on Mode A reached 16.7pp — larger than several of the reported
    effects — so a single-seed row cannot be presented as a condition estimate.
    Report `spread` next to any effect it exceeds.
    """
    piv = df.groupby(["condition", "seed"])[metric].mean().unstack()
    piv.columns = [f"seed_{c}" for c in piv.columns]
    piv["mean"] = piv.mean(axis=1)
    piv["spread"] = piv.filter(like="seed_").max(axis=1) - \
        piv.filter(like="seed_").min(axis=1)
    piv["n_seeds"] = df.groupby("condition")["seed"].nunique()
    return piv.round(4)


def holm_correct(results: list[dict]) -> list[dict]:
    """Holm-Bonferroni over a DECLARED family of contrast results.

    The family is whatever list you pass. Declare it explicitly in Ch3 and keep
    Mode A, Mode B and the logit coefficients in separate declared families —
    do not silently omit a contrast from the family it belongs to.
    """
    m = len(results)
    order = sorted(range(m), key=lambda i: results[i]["p"])
    out = [dict(r) for r in results]
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * results[i]["p"])
        running_max = max(running_max, adj)
        out[i]["p_holm"] = running_max
        out[i]["significant_05"] = running_max < 0.05
    return out


def logistic_regression(df: pd.DataFrame, extra_terms: list[str] | None = None,
                        cluster: str | None = None):
    """success ~ C(condition, Treatment('b')) [+ extras]. Returns fitted model.

    `cluster` (e.g. "episode_id") switches to cluster-robust standard errors.
    Mode B queries from the same episode share a history and are not
    independent; plain SEs are too narrow.
    """
    import statsmodels.formula.api as smf

    terms = ["C(condition, Treatment('b'))"] + (extra_terms or [])
    formula = "success ~ " + " + ".join(terms)
    model = smf.logit(formula, data=df)
    if cluster and cluster in df.columns:
        return model.fit(disp=False, cov_type="cluster",
                         cov_kwds={"groups": df[cluster]})
    return model.fit(disp=False)


def odds_ratio_table(model, alpha: float = 0.05) -> pd.DataFrame:
    """Coefficients as odds ratios with CIs — the effect size Ch4 is missing.

    Ch4 currently reports bare log-odds/mean differences. An examiner will ask
    for the effect size and its interval, not just the p-value.
    """
    ci = model.conf_int(alpha=alpha)
    out = pd.DataFrame({
        "coef": model.params,
        "odds_ratio": np.exp(model.params),
        "or_ci_low": np.exp(ci[0]),
        "or_ci_high": np.exp(ci[1]),
        "p": model.pvalues,
    })
    return out.round(4)


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a single proportion.

    Preferred over the normal approximation at the rates in this study: Mode A
    baseline success is ~0.03, where a Wald interval can cross zero.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = sps.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def failure_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["condition"], df["failure_tag"], normalize="index").round(3)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Headline per-condition table: success (with Wilson CI), recall, cost.

    Cost columns: `prompt_tokens` is the ANSWERING call only and understates any
    condition with a write path — quoting it as total cost is what produced the
    withdrawn 45% saving. `total_prompt_tokens` (P3-metered) includes extraction
    and write calls and is the column Ch3's definition actually refers to. Both
    are shown so the gap is visible rather than hidden.
    """
    spec = {
        "n": ("success", "size"),
        "success_rate": ("success", "mean"),
        "evidence_recall": ("evidence_recall", "mean"),
        "prompt_tokens_answer_only": ("prompt_tokens", "mean"),
        "completion_tokens": ("completion_tokens", "mean"),
        "memory_written_tokens": ("memory_written_tokens", "mean"),
        "overhead_calls": ("overhead_calls", "mean"),
        "latency_ms": ("latency_ms", "mean"),
    }
    if "total_prompt_tokens" in df.columns:
        spec["total_prompt_tokens"] = ("total_prompt_tokens", "mean")
        spec["total_llm_calls"] = ("total_llm_calls", "mean")
    spec = {k: v for k, v in spec.items() if v[0] in df.columns}
    agg = df.groupby("condition").agg(**spec)

    ci = df.groupby("condition")["success"].agg(["sum", "size"]).apply(
        lambda r: wilson_ci(int(r["sum"]), int(r["size"])), axis=1)
    agg["success_ci_low"] = [c[0] for c in ci]
    agg["success_ci_high"] = [c[1] for c in ci]
    return agg.round(3).sort_index()
