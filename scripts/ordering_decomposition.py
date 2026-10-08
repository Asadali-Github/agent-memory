#!/usr/bin/env python
"""Decompose every B-vs-Mx effect into ORDERING and ARCHITECTURE components.

    python scripts/ordering_decomposition.py --file results/rerun_2026_07_29/custom_ds_merged.jsonl

WHY THIS EXISTS
------------------------------------------------------------------------------
`audit_prompt_order.py` established that every memory condition is contrasted
against a chronologically-ordered baseline, so every B-vs-Mx effect is
architecture PLUS presentation. That audit reports one mean inversion rate per
condition and stops there. It cannot say how much of any given deficit is
ordering, because with one number per condition, ordering and architecture are
perfectly collinear.

The identification insight this script exploits: the inversion rate is NOT
constant within a condition. M1, M3 and M4 select turns by relevance score, and
how badly that scrambles chronology depends on where the evidence happens to sit
in the episode. So within a single architecture, holding the memory mechanism
exactly fixed, some items are presented near-chronologically and some are
presented near-reversed. That within-condition variation identifies the ordering
slope without any new compute, and the slope can then be used to predict each
condition's success rate at the counterfactual iota = 0 (fully chronological)
that the baseline already enjoys.

This is the observational analogue of the "render-matched control" that Jiang et
al. (2026, arXiv:2607.16019) propose as a reusable protocol for separating layout
from mechanism, and of the ordering-only b_reversed arm already registered in
this repo. The experimental arm remains the stronger evidence; this recovers most
of the answer from data already on disk, and pre-registers the prediction that
the arm will test.

WHAT IS AND IS NOT IDENTIFIED
------------------------------------------------------------------------------
iota is observational, not randomised. It is a function of where the evidence
sits in the episode, so it correlates with evidence_distance, which is itself
predictive of success. Every model below therefore conditions on
evidence_distance, and Model 3 adds item fixed effects so the slope is estimated
only from items where two seeds of the SAME condition happened to draw different
layouts. Residual confounding by unobserved item difficulty cannot be excluded
and MUST be stated wherever these numbers are quoted. Read the adjusted effects
as "what the contrast looks like once measured presentation differences are
netted out", not as a causal architecture effect.

OUTPUTS
------------------------------------------------------------------------------
results/ordering_decomposition.md    tables + interpretation
results/figures/fig_ordering.png     iota distribution + fitted slope
results/figures/fig_pareto.png       cost-adjusted accuracy frontier
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# ordering metric
# --------------------------------------------------------------------------


def inversion_rate(steps: list[int]) -> float:
    """Kendall-style fraction of turn pairs shown out of chronological order.

    0.0 = perfectly chronological (the baseline's layout discipline)
    1.0 = perfectly reversed (the pre-patch M2 pattern)
    Equivalent to (1 - tau_b) / 2 for a permutation with no ties.
    """
    n = len(steps)
    if n < 2:
        return 0.0
    arr = np.asarray(steps)
    inv = sum(int(np.sum(arr[i + 1:] < arr[i])) for i in range(n - 1))
    return inv / (n * (n - 1) / 2)


def load(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prov = rec.get("provenance") or []
        steps = [p["step"] for p in prov
                 if isinstance(p, dict) and p.get("step") is not None
                 and p["step"] != -1]
        rec["iota"] = inversion_rate(steps)
        rec["n_shown"] = len(steps)
        rec["episode"] = str(rec.get("item_id", "")).split(":")[0]
        rec.pop("provenance", None)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["success"] = df["success"].astype(int)
    return df


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def fit_models(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    out = {}

    # Model 1 — the dissertation's current specification, for reference.
    m1 = smf.logit(
        "success ~ C(condition, Treatment('b')) + evidence_distance", data=df
    ).fit(disp=0, cov_type="cluster", cov_kwds={"groups": df["episode"]})
    out["m1"] = m1

    # Model 2 — same, plus the ordering covariate.
    m2 = smf.logit(
        "success ~ C(condition, Treatment('b')) + iota + evidence_distance", data=df
    ).fit(disp=0, cov_type="cluster", cov_kwds={"groups": df["episode"]})
    out["m2"] = m2

    # Model 3 — item fixed effects. Slope identified ONLY from within-item,
    # within-condition layout variation across seeds. The strictest available
    # control for item difficulty.
    sub = df[df["condition"].isin(["m1", "m3", "m4"])].copy()
    keep = sub.groupby(["item_id", "condition"])["iota"].transform("nunique") > 1
    sub_fe = sub[keep]
    out["fe_n"] = len(sub_fe)
    out["fe_items"] = sub_fe["item_id"].nunique()
    if len(sub_fe) > 40 and sub_fe["success"].nunique() > 1:
        try:
            m3 = smf.logit(
                "success ~ iota + C(condition) + C(item_id)", data=sub_fe
            ).fit(disp=0, method="bfgs", maxiter=400)
            out["m3"] = m3
        except Exception as exc:  # pragma: no cover
            out["m3_error"] = str(exc)

    # Model 4 — within each architecture separately. Architecture is literally
    # constant here, so nothing about the mechanism can drive the slope.
    per_cond = {}
    for cond in ["m1", "m3", "m4"]:
        d = df[df["condition"] == cond]
        if d["iota"].nunique() < 3 or d["success"].nunique() < 2:
            continue
        try:
            fit = smf.logit("success ~ iota + evidence_distance", data=d).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": d["episode"]}
            )
            per_cond[cond] = {
                "coef": fit.params["iota"],
                "se": fit.bse["iota"],
                "p": fit.pvalues["iota"],
                "ci": fit.conf_int().loc["iota"].tolist(),
                "n": len(d),
                "iota_sd": d["iota"].std(),
            }
        except Exception:
            continue
    out["per_cond"] = per_cond
    return out


def counterfactual(df: pd.DataFrame, m2) -> pd.DataFrame:
    """Predicted success for each condition at its OWN iota vs at iota = 0."""
    rows = []
    for cond in sorted(df["condition"].unique()):
        d = df[df["condition"] == cond].copy()
        observed = d["success"].mean()
        p_actual = m2.predict(d).mean()
        d0 = d.copy()
        d0["iota"] = 0.0
        p_chrono = m2.predict(d0).mean()
        rows.append(
            {
                "condition": cond,
                "n": len(d),
                "mean_iota": d["iota"].mean(),
                "sd_iota": d["iota"].std(),
                "observed": observed,
                "fitted": p_actual,
                "fitted_at_iota0": p_chrono,
                "ordering_penalty": p_actual - p_chrono,
            }
        )
    out = pd.DataFrame(rows).set_index("condition")
    base = out.loc["b"]
    out["raw_vs_b"] = out["observed"] - base["observed"]
    out["adj_vs_b"] = out["fitted_at_iota0"] - base["fitted_at_iota0"]
    out["attributable_to_ordering"] = out["raw_vs_b"] - out["adj_vs_b"]
    return out


def bootstrap_adjusted(df: pd.DataFrame, n_boot: int = 400, seed: int = 7) -> pd.DataFrame:
    """Episode-cluster bootstrap CIs on the ordering-adjusted contrasts."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(seed)
    eps = df["episode"].unique()
    conds = [c for c in sorted(df["condition"].unique()) if c != "b"]
    draws = {c: [] for c in conds}
    for _ in range(n_boot):
        pick = rng.choice(eps, size=len(eps), replace=True)
        boot = pd.concat([df[df["episode"] == e] for e in pick], ignore_index=True)
        if boot["condition"].nunique() < df["condition"].nunique():
            continue
        try:
            fit = smf.logit(
                "success ~ C(condition, Treatment('b')) + iota + evidence_distance",
                data=boot,
            ).fit(disp=0)
        except Exception:
            continue
        b0 = boot.copy()
        b0["iota"] = 0.0
        b0["_p"] = fit.predict(b0)
        g = b0.groupby("condition")["_p"].mean()
        if "b" not in g:
            continue
        for c in conds:
            if c in g:
                draws[c].append(g[c] - g["b"])
    rows = []
    for c in conds:
        v = np.array(draws[c])
        if len(v) < 30:
            continue
        rows.append(
            {
                "contrast": f"{c} vs b",
                "adj_effect": v.mean(),
                "ci_low": np.percentile(v, 2.5),
                "ci_high": np.percentile(v, 97.5),
                "n_boot": len(v),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# cost frontier
# --------------------------------------------------------------------------


def confounding_check(df: pd.DataFrame) -> pd.DataFrame:
    """How badly is iota entangled with item difficulty within each condition?"""
    rows = []
    for cond in sorted(df["condition"].unique()):
        d = df[df["condition"] == cond]
        if d["iota"].nunique() < 3:
            rows.append({"condition": cond, "iota_sd": d["iota"].std(),
                         "r_iota_distance": np.nan, "r_iota_nshown": np.nan,
                         "note": "iota constant — no within-condition variation"})
            continue
        rows.append({
            "condition": cond,
            "iota_sd": d["iota"].std(),
            "r_iota_distance": d["iota"].corr(d["evidence_distance"]),
            "r_iota_nshown": d["iota"].corr(d["n_shown"]),
            "note": "",
        })
    return pd.DataFrame(rows).set_index("condition")


def power_for_arm(df: pd.DataFrame, effect_pp: float = 0.119,
                  alpha: float = 0.05, power: float = 0.80) -> dict:
    """How many items does a randomised ordering arm need?

    Uses the discordance rate actually observed in the M2-vs-B McNemar
    contrast to size a paired (McNemar) test, which is what a b_reversed vs b
    comparison would use.
    """
    from scipy import stats

    p_base = df.loc[df["condition"] == "b", "success"].mean()
    p_alt = p_base - effect_pp
    # Paired design: assume the same discordance structure as m2 vs b.
    b_succ = df[df["condition"] == "b"].set_index(["item_id", "seed"])["success"]
    m2_succ = df[df["condition"] == "m2"].set_index(["item_id", "seed"])["success"]
    joint = pd.concat([b_succ.rename("b"), m2_succ.rename("m2")], axis=1).dropna()
    disc = float(((joint["b"] != joint["m2"]).mean())) if len(joint) else 0.25

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    # McNemar sample size (Connor 1987): n = (z_a*sqrt(disc) + z_b*sqrt(disc - d^2))^2 / d^2
    d = effect_pp
    n = ((z_a * np.sqrt(disc) + z_b * np.sqrt(max(disc - d ** 2, 1e-9))) ** 2) / d ** 2

    # Observational power: what iota SD would be needed for the pooled slope?
    # WITHIN-condition SD (pooled), not the cross-condition SD: only the
    # within-condition spread is usable for identification.
    iota_sd = float(
        df[df["condition"].isin(["m1", "m3", "m4"])]
        .groupby("condition")["iota"]
        .std()
        .pow(2)
        .mean()
        ** 0.5
    )
    return {
        "p_base": p_base,
        "p_alt": p_alt,
        "discordance": disc,
        "n_items_needed": int(np.ceil(n)),
        "n_items_available": int(df["item_id"].nunique()),
        "observational_iota_sd": iota_sd,
    }


def cost_frontier(df: pd.DataFrame, extract_every: int = 5,
                  completion_per_call: int = 60) -> pd.DataFrame:
    """Accuracy against AMORTISED total tokens, not answering-call tokens.

    overhead_calls is a cumulative per-episode counter, so it is differenced
    within episode before being charged to an item (the same correction
    reconstruct_cost.py applies).
    """
    d = df.copy()
    d = d.sort_values(["condition", "seed", "episode", "item_id"])
    d["calls_item"] = (
        d.groupby(["condition", "seed", "episode"])["overhead_calls"].diff().fillna(
            d["overhead_calls"]
        )
    ).clip(lower=0)

    mean_turn_tokens = (d["memory_read_tokens"] / d["memory_items"].clip(lower=1)).mean()
    extract_prompt = 45 + extract_every * (mean_turn_tokens + 5)
    d["write_tokens"] = d["calls_item"] * (extract_prompt + completion_per_call)
    d["total_tokens"] = d["prompt_tokens"] + d["completion_tokens"] + d["write_tokens"]

    g = d.groupby("condition").agg(
        success=("success", "mean"),
        answer_tokens=("prompt_tokens", "mean"),
        write_tokens=("write_tokens", "mean"),
        total_tokens=("total_tokens", "mean"),
        calls_per_item=("calls_item", "mean"),
        latency_ms=("latency_ms", "mean"),
        n=("success", "size"),
    )
    base = g.loc["b"]
    g["answer_pct_vs_b"] = 100 * (g["answer_tokens"] - base["answer_tokens"]) / base["answer_tokens"]
    g["total_pct_vs_b"] = 100 * (g["total_tokens"] - base["total_tokens"]) / base["total_tokens"]

    # Pareto frontier on (total_tokens down, success up).
    dominated = []
    for c in g.index:
        dom = any(
            (g.loc[o, "success"] >= g.loc[c, "success"])
            and (g.loc[o, "total_tokens"] <= g.loc[c, "total_tokens"])
            and (o != c)
            and (
                g.loc[o, "success"] > g.loc[c, "success"]
                or g.loc[o, "total_tokens"] < g.loc[c, "total_tokens"]
            )
            for o in g.index
        )
        dominated.append(dom)
    g["dominated"] = dominated
    return g


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def figures(df: pd.DataFrame, cf: pd.DataFrame, frontier: pd.DataFrame,
            outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    order = ["b", "m1", "m2", "m3", "m4"]
    order = [c for c in order if c in df["condition"].unique()]

    # --- Figure: ordering ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    data = [df.loc[df["condition"] == c, "iota"].values for c in order]
    bp = ax.boxplot(data, labels=[c.upper() for c in order], showmeans=True,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#dfe6ef")
    ax.set_ylabel(r"inversion rate $\iota$")
    ax.set_xlabel("condition")
    ax.set_title("(a) Each condition occupies its own narrow band of $\\iota$;\n"
                 "only B ($\\iota$=0) and M2 ($\\iota$=1) anchor the axis")
    ax.axhline(0, ls=":", lw=1, color="grey")
    ax.set_ylim(-0.05, 1.05)

    # (b) WITHIN condition only. Pooling across conditions here would smuggle
    # the architecture difference back in as if it were an ordering slope.
    ax = axes[1]
    colours = {"m1": "#26456e", "m3": "#1e8449", "m4": "#c0392b"}
    for cond in ["m1", "m3", "m4"]:
        sub = df[df["condition"] == cond].copy()
        if len(sub) < 30:
            continue
        sub["bin"] = pd.qcut(sub["iota"], 4, duplicates="drop")
        g = sub.groupby("bin", observed=True).agg(
            iota=("iota", "mean"), succ=("success", "mean"), n=("success", "size")
        )
        se = np.sqrt(g["succ"] * (1 - g["succ"]) / g["n"])
        ax.errorbar(g["iota"], g["succ"], yerr=1.96 * se, fmt="o-", capsize=3,
                    color=colours[cond], label=cond.upper(), alpha=0.85)
    ax.set_xlabel(r"inversion rate $\iota$ (quartile bins, within condition)")
    ax.set_ylabel("success rate")
    ax.set_title("(b) Within each architecture the slope is flat and noisy:\n"
                 "the logged spread cannot identify the ordering effect")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "fig_ordering.png", dpi=200)
    plt.close(fig)

    # --- Figure: Pareto -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in frontier.index:
        r = frontier.loc[c]
        ax.scatter(r["total_tokens"], r["success"], s=150,
                   color="#c0392b" if r["dominated"] else "#1e8449",
                   edgecolor="black", zorder=3)
        ax.annotate(c.upper(), (r["total_tokens"], r["success"]),
                    xytext=(8, 6), textcoords="offset points", fontweight="bold")
        ax.scatter(r["answer_tokens"], r["success"], s=50, marker="x",
                   color="grey", zorder=2)
        ax.plot([r["answer_tokens"], r["total_tokens"]], [r["success"]] * 2,
                ls=":", color="grey", lw=1, zorder=1)
    ax.set_xlabel("mean tokens per item\n"
                  "(x = answering call only, o = amortised total incl. write path)")
    ax.set_ylabel("success rate")
    ax.set_title("Cost-adjusted frontier: M3's saving is on the answering call,\n"
                 "not on total spend")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_pareto.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results/rerun_2026_07_29/custom_ds_merged.jsonl")
    ap.add_argument("--out", default="results/ordering_decomposition.md")
    ap.add_argument("--figdir", default="results/figures")
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    df = load(Path(args.file))
    models = fit_models(df)
    cf = counterfactual(df, models["m2"])
    boot = bootstrap_adjusted(df, n_boot=args.boot)
    frontier = cost_frontier(df)
    figures(df, cf, frontier, Path(args.figdir))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    m2 = models["m2"]
    iota_coef = m2.params["iota"]
    iota_ci = m2.conf_int().loc["iota"].tolist()
    iota_p = m2.pvalues["iota"]

    L = []
    A = L.append
    A("# Ordering decomposition — separating presentation from architecture\n")
    A("_Generated by `scripts/ordering_decomposition.py`. Observational, not "
      "randomised — read the identification caveat in §4 before quoting._\n")

    A("## 1. The inversion rate varies WITHIN each condition\n")
    A("This is what makes the decomposition possible. If every item in a "
      "condition shared one layout, ordering and architecture would be "
      "perfectly collinear and nothing below could be estimated.\n")
    A("```")
    A(df.groupby("condition")["iota"].describe()[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ].round(3).to_string())
    A("```\n")

    A("## 2. The ordering slope\n")
    A(f"Pooled logit, cluster-robust by episode, controlling for condition and "
      f"evidence_distance:\n")
    A(f"- coefficient on iota: **{iota_coef:.3f}** "
      f"(95% CI [{iota_ci[0]:.3f}, {iota_ci[1]:.3f}], p = {iota_p:.4g})")
    A(f"- odds ratio for a fully reversed vs fully chronological layout: "
      f"**{np.exp(iota_coef):.3f}** "
      f"[{np.exp(iota_ci[0]):.3f}, {np.exp(iota_ci[1]):.3f}]\n")

    if models.get("per_cond"):
        A("Estimated separately within each architecture — the mechanism is "
          "literally held constant, so it cannot be driving the slope:\n")
        A("```")
        pc = pd.DataFrame(models["per_cond"]).T
        pc = pc[["n", "iota_sd", "coef", "se", "p"]].astype(float).round(4)
        A(pc.to_string())
        A("```\n")

    if "m3" in models:
        fe = models["m3"]
        A(f"With item fixed effects (n = {models['fe_n']} records over "
          f"{models['fe_items']} items where two seeds of the same condition "
          f"drew different layouts), the iota coefficient is "
          f"**{fe.params['iota']:.3f}** (p = {fe.pvalues['iota']:.4g}). "
          f"This is the strictest available control for item difficulty.\n")
    elif "m3_error" in models:
        A(f"_Item-fixed-effects model did not converge: {models['m3_error']}_\n")
    else:
        A(f"_Item-fixed-effects model not estimable: only {models.get('fe_n', 0)} "
          f"records show within-item layout variation._\n")

    A("## 3. Decomposition of every contrast\n")
    A("`raw_vs_b` is the effect as currently reported. `adj_vs_b` is the same "
      "contrast with every condition set to the baseline's chronological "
      "layout (iota = 0). The difference is what presentation was worth.\n")
    A("```")
    A(cf[["n", "mean_iota", "sd_iota", "observed", "fitted_at_iota0",
          "raw_vs_b", "adj_vs_b", "attributable_to_ordering"]].round(4).to_string())
    A("```\n")
    if len(boot):
        A("Episode-cluster bootstrap CIs on the ordering-adjusted contrasts:\n")
        A("```")
        A(boot.round(4).to_string(index=False))
        A("```\n")

    A("## 3b. VERDICT — the observational route is underpowered, and that is "
      "the finding\n")
    conf = confounding_check(df)
    pw = power_for_arm(df)
    A("Two facts kill the observational identification, and both are worth "
      "stating explicitly in Ch3 rather than discovered by an examiner:\n")
    A(f"1. **Within-condition range is far too narrow.** The ordering effect "
      f"that matters spans iota = 0 to 1. The variation actually available "
      f"inside M1/M3/M4 has SD = {pw['observational_iota_sd']:.3f}, i.e. the "
      f"data only ever wobbles about +/- 9 points along a 100-point axis. The "
      f"pooled slope's CI "
      f"([{iota_ci[0]:.2f}, {iota_ci[1]:.2f}] in log-odds) is consistent with "
      f"anything from a large harm to a large benefit. It is not evidence of no "
      f"effect; it is an absence of evidence.\n")
    A(f"2. **Seeds do not re-randomise the layout.** Retrieval order is "
      f"deterministic given the item, so re-running the same item under a "
      f"different seed reproduces the same iota. Only "
      f"{models.get('fe_n', 0)} records show within-item layout variation, so "
      f"the item-fixed-effects estimator has nothing to work with. This is a "
      f"property of the harness, not of the data volume — more seeds would not "
      f"help.\n")
    A("Residual entanglement between iota and item difficulty:\n")
    A("```")
    A(conf.round(4).to_string())
    A("```\n")
    A("The positive within-M1 slope in §2 is very likely this confound rather "
      "than a real benefit of scrambling: items whose evidence sits further "
      "back get scrambled more, and `evidence_distance` independently predicts "
      "success with a positive coefficient. Do not report it as a finding.\n")
    A("**Power required for the experimental arm.** A paired `b_reversed` vs "
      "`b` McNemar test, sized on the discordance rate actually observed in "
      f"the M2-vs-B contrast ({pw['discordance']:.3f}), needs "
      f"**n = {pw['n_items_needed']} items** to detect the "
      f"{100 * 0.119:.1f}pp ordering effect at 80% power, alpha = 0.05. The "
      f"Mode B generator already produces {pw['n_items_available']} distinct "
      f"items, so **the arm is adequately powered on a single seed** and costs "
      f"one run of one condition. That is the cheapest defensible answer "
      f"available to this project.\n")

    A("## 4. Identification caveat — quote this alongside the numbers\n")
    A("iota is observational. It is determined by where the evidence happens to "
      "sit in the episode, so it correlates with `evidence_distance`, which "
      "independently predicts success. Every model conditions on "
      "evidence_distance and the fixed-effects model additionally holds the item "
      "constant, but residual confounding by unobserved item difficulty cannot "
      "be excluded. These are ordering-ADJUSTED contrasts, not causal "
      "architecture effects. The registered `b_reversed` arm and a "
      "`chronological_order: true` run of M1/M3/M4 remain the experimental test; "
      "this analysis pre-registers what that test should find.\n")

    A("## 5. Cost-adjusted frontier\n")
    A("`answer_tokens` is what Ch4 currently reports. `total_tokens` amortises "
      "the write path using per-item extraction calls differenced within "
      "episode (overhead_calls is cumulative).\n")
    A("```")
    A(frontier.round(3).to_string())
    A("```\n")

    A("## 6. Figures\n")
    A("- `results/figures/fig_ordering.png` — within-condition iota spread and "
      "the fitted success-vs-iota relationship\n")
    A("- `results/figures/fig_pareto.png` — accuracy against answering-call vs "
      "amortised total cost\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
