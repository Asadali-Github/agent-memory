#!/usr/bin/env python
"""Chapter 4 confirmatory statistics — regenerated from the merged run records.

    python scripts/ch4_stats.py

Data (post-fix, canonical per results/audit_2026_08_02.md):
  Mode B: results/rerun_2026_07_29/custom_ds_merged.jsonl   (1,971 records, 9 arms x 219)
  Mode A: results/rerun_2026_07_29/longmemeval_s_merged.jsonl (1,800 records)

Tests:
  * exact McNemar on paired per-item success, Holm-corrected across the declared
    family of 15 contrasts (Ch3 SS stats protocol);
  * episode-clustered bootstrap 95% CIs (4000 resamples) for success rates and
    risk differences (Mode B clusters = (seed, episode); Mode A clusters = items);
  * Wilcoxon signed-rank STRICTLY for the continuous efficiency metrics
    (total tokens, latency), Holm-corrected within each metric;
  * logistic regression of Mode B success on arm + evidence distance with
    episode-cluster-robust SEs.
"""

from __future__ import annotations

import collections
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
MODE_B = ROOT / "results" / "rerun_2026_07_29" / "custom_ds_merged.jsonl"
MODE_A = ROOT / "results" / "rerun_2026_07_29" / "longmemeval_s_merged.jsonl"

BASE_SEED = 20260802
B_RESAMPLES = 4000

B_ARMS = ["b", "m1", "m2", "m3", "m4",
          "b_reversed", "m1_chrono", "m3_chrono", "m4_chrono"]
A_ARMS = ["b", "m1", "m2", "m3", "m4"]

# Declared confirmatory family (contrast arm, reference arm, mode).
CONTRASTS = [
    ("m1", "b", "A"), ("m4", "b", "A"), ("m3", "b", "A"), ("m2", "b", "A"),
    ("m4_chrono", "m4", "B"), ("m1_chrono", "b", "B"), ("m4_chrono", "b", "B"),
    ("m1_chrono", "m1", "B"), ("b_reversed", "b", "B"), ("m2", "b", "B"),
    ("m1", "b", "B"), ("m3_chrono", "m3", "B"), ("m3", "b", "B"),
    ("m3_chrono", "b", "B"), ("m4", "b", "B"),
]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def episode(rec):
    return rec["item_id"].split(":")[0]


def total_tokens(rec):
    tp, tc = rec.get("total_prompt_tokens"), rec.get("total_completion_tokens")
    if tp is None or tc is None:
        return None
    return tp + tc


# ---------------------------------------------------------------- bootstrap
def _label_seed(label: str) -> int:
    """Stable per-contrast seed: base seed mixed with a CRC of the label.

    Each interval is drawn from its own generator seeded only by BASE_SEED and
    the contrast label, so an interval does not depend on how many intervals
    were computed before it. Using one module-level generator for every call
    (as an earlier version did) makes each CI a function of call ORDER, which
    means adding or reordering a contrast silently shifts every later interval
    and the printed bounds cannot be reproduced by re-running the script.
    """
    return (BASE_SEED + zlib.crc32(label.encode("utf-8"))) % (2 ** 32)


def clustered_boot(values, clusters, label, B=B_RESAMPLES):
    """Percentile 95% CI of the mean, resampling whole clusters.

    `label` identifies the estimand (arm or contrast name) and seeds this
    interval's own generator — see _label_seed.
    """
    rng = np.random.default_rng(_label_seed(label))
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    by_cluster = {c: values[clusters == c] for c in uniq}
    n = len(uniq)
    stats = np.empty(B)
    for i in range(B):
        draw = rng.choice(uniq, size=n, replace=True)
        pooled = np.concatenate([by_cluster[c] for c in draw])
        stats[i] = pooled.mean()
    return np.percentile(stats, [2.5, 97.5])


def paired_boot_ci(diffs, clusters, label, B=B_RESAMPLES):
    return clustered_boot(np.asarray(diffs, dtype=float),
                          np.asarray(clusters), label, B)


# ------------------------------------------------------------------ pairing
def index_by_key(recs):
    return {(r["seed"], r["item_id"]): r for r in recs}


def pair_outcomes(ix, iy):
    keys = sorted(set(ix) & set(iy))
    xs = np.array([ix[k]["success"] for k in keys], dtype=float)
    ys = np.array([iy[k]["success"] for k in keys], dtype=float)
    return keys, xs, ys


def mcnemar(xs, ys):
    n10 = int(np.sum((xs == 1) & (ys == 0)))  # contrast-only correct
    n01 = int(np.sum((xs == 0) & (ys == 1)))  # reference-only correct
    n = n10 + n01
    if n == 0:
        return n10, n01, np.nan, 1.0
    p = sps.binomtest(min(n10, n01), n, 0.5).pvalue
    oratio = np.inf if n01 == 0 else n10 / n01
    return n10, n01, oratio, p


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


# --------------------------------------------------------------------- main
def main():
    b_recs = load(MODE_B)
    a_recs = load(MODE_A)
    print(f"Loaded Mode B: {len(b_recs)} records; Mode A: {len(a_recs)} records")

    b_by_arm = collections.defaultdict(list)
    for r in b_recs:
        b_by_arm[r["condition"]].append(r)
    a_by_arm = collections.defaultdict(list)
    for r in a_recs:
        a_by_arm[r["condition"]].append(r)

    # ---- 1. headline tables -------------------------------------------------
    print("\n=== 1. Mode B headline (n per arm, success [cluster-boot CI], recall, "
          "answering tokens, latency p50/p95) ===")
    for c in B_ARMS:
        rs = b_by_arm[c]
        succ = np.array([r["success"] for r in rs], dtype=float)
        cl = np.array([(r["seed"], episode(r)) for r in rs], dtype=object)
        cl = np.array([f"{s}|{e}" for s, e in cl])
        lo, hi = clustered_boot(succ, cl, f"B|{c}")
        rec = np.mean([r["evidence_recall"] for r in rs])
        tok = np.mean([r["prompt_tokens"] for r in rs])
        lat = np.percentile([r["latency_ms"] for r in rs], [50, 95])
        print(f"  {c:12s} n={len(rs)} succ={succ.mean():.4f} "
              f"[{lo:.3f},{hi:.3f}] recall={rec:.3f} tok={tok:.0f} "
              f"lat50={lat[0]:.0f} lat95={lat[1]:.0f}")

    print("\n=== 1b. Mode A headline (recall: undefined on abstention items, "
          "conservatively counted 0; n_recall = defined records) ===")
    for c in A_ARMS:
        rs = a_by_arm[c]
        succ = np.array([r["success"] for r in rs], dtype=float)
        # cluster on item_id alone: the same question may appear under several
        # seeds (seeded cap-120 draws overlap), and repeated draws are not
        # independent trials of the item.
        cl = np.array([r["item_id"] for r in rs])
        lo, hi = clustered_boot(succ, cl, f"A|{c}")
        rec_defined = [r["evidence_recall"] for r in rs
                       if r.get("evidence_recall") is not None]
        rec_zero = np.sum(rec_defined) / len(rs)
        tok = np.mean([r["prompt_tokens"] for r in rs])
        lat = np.percentile([r["latency_ms"] for r in rs], 50)
        print(f"  {c:4s} n={len(rs)} succ={succ.mean():.4f} [{lo:.3f},{hi:.3f}] "
              f"recall0={rec_zero:.3f} (defined-only mean="
              f"{np.mean(rec_defined):.3f} on {len(rec_defined)}) "
              f"tok={tok:.0f} lat50={lat:.0f}")

    # ---- 2. McNemar battery + Holm -----------------------------------------
    print("\n=== 2. Registered confirmatory family: exact McNemar, Holm over 15 ===")
    b_idx = {c: index_by_key(b_by_arm[c]) for c in B_ARMS}
    a_idx = {c: index_by_key(a_by_arm[c]) for c in A_ARMS}
    rows = []
    for x, y, mode in CONTRASTS:
        idx = b_idx if mode == "B" else a_idx
        keys, xs, ys = pair_outcomes(idx[x], idx[y])
        n10, n01, oratio, p = mcnemar(xs, ys)
        rd = float(np.mean(xs - ys))
        if mode == "B":
            cl = np.array([f"{k[0]}|{k[1].split(':')[0]}" for k in keys])
        else:
            cl = np.array([k[1] for k in keys])  # cluster on item identity
        lo, hi = paired_boot_ci(xs - ys, cl, f"{mode}|{x}|{y}")
        rows.append(dict(mode=mode, contrast=f"{x} vs {y}", n=len(keys),
                         n10=n10, n01=n01, rd=rd, lo=lo, hi=hi,
                         oratio=oratio, p=p))
    adj = holm([r["p"] for r in rows])
    for r, pa in zip(rows, adj):
        r["p_holm"] = pa
        print(f"  {r['mode']} {r['contrast']:22s} n={r['n']} disc={r['n10']}/{r['n01']} "
              f"RD={r['rd']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
              f"OR={r['oratio']:.2f} p={r['p']:.3e} p_Holm={pa:.3e}")
    print(f"  significant after Holm: {sum(r['p_holm'] < 0.05 for r in rows)}/15")

    # ---- 3. Wilcoxon on continuous efficiency metrics -----------------------
    print("\n=== 3. Wilcoxon signed-rank (continuous metrics only), Holm within metric ===")
    for mode, idx, arms in (("B", b_idx, B_ARMS), ("A", a_idx, A_ARMS)):
        for metric, getter in (("total_tokens", total_tokens),
                               ("latency_ms", lambda r: r.get("latency_ms"))):
            fam = []
            for c in arms:
                if c == "b":
                    continue
                keys = sorted(set(idx[c]) & set(idx["b"]))
                pairs = [(getter(idx[c][k]), getter(idx["b"][k])) for k in keys]
                pairs = [(u, v) for u, v in pairs if u is not None and v is not None]
                diffs = np.array([u - v for u, v in pairs], dtype=float)
                nz = diffs[diffs != 0]
                if len(nz) < 10:
                    fam.append((f"{c} vs b", len(pairs), np.nan))
                    continue
                w = sps.wilcoxon(diffs, zero_method="wilcox",
                                 alternative="two-sided")
                fam.append((f"{c} vs b", len(pairs), float(np.median(diffs)),
                            float(np.mean(diffs)), w.pvalue))
            adj = holm([f[4] for f in fam])
            for (name, n, med, mean, p), pa in zip(fam, adj):
                print(f"  Mode {mode} {metric:13s} {name:18s} n={n} "
                      f"meddiff={med:+.1f} meandiff={mean:+.1f} "
                      f"p={p:.3e} p_Holm={pa:.3e}")

    # ---- 4. depth sensitivity (Mode B) --------------------------------------
    print("\n=== 4. Mode B depth sensitivity (min evidence_distance in panel = "
          f"{min(r['evidence_distance'] for r in b_recs)}) ===")
    for thr in (20, 40, 60, 80):
        line = [f"d>={thr:3d}"]
        for c in ("b", "m1", "m2", "m4", "m1_chrono", "m4_chrono"):
            rs = [r for r in b_by_arm[c] if r["evidence_distance"] >= thr]
            line.append(f"{c}={np.mean([r['success'] for r in rs]):.3f}(n={len(rs)})")
        print("  " + "  ".join(line))
        # key paired contrasts within the subset
        for x, y in (("b_reversed", "b"), ("m1_chrono", "b"), ("m1_chrono", "m1"),
                     ("m1", "b"), ("m4_chrono", "m4")):
            ix = {k: r for k, r in b_idx[x].items() if r["evidence_distance"] >= thr}
            iy = {k: r for k, r in b_idx[y].items() if r["evidence_distance"] >= thr}
            keys, xs, ys = pair_outcomes(ix, iy)
            n10, n01, _, p = mcnemar(xs, ys)
            print(f"      {x} vs {y}: RD={np.mean(xs-ys):+.3f} "
                  f"disc={n10}/{n01} p={p:.2e}")

    # ---- 5. logistic regression (Mode B) ------------------------------------
    print("\n=== 5. Logistic regression, Mode B (cluster-robust by seed x episode) ===")
    df = pd.DataFrame(dict(
        success=[int(r["success"]) for r in b_recs],
        condition=pd.Categorical([r["condition"] for r in b_recs],
                                 categories=B_ARMS),
        dist=[r["evidence_distance"] for r in b_recs],
        cluster=[f"{r['seed']}|{episode(r)}" for r in b_recs],
    ))
    model = smf.logit("success ~ C(condition, Treatment(reference='b')) + dist",
                      data=df).fit(disp=0, cov_type="cluster",
                                   cov_kwds={"groups": df["cluster"]})
    ci = model.conf_int()
    for name in model.params.index:
        print(f"  {name:45s} {model.params[name]:+.3f} "
              f"p={model.pvalues[name]:.4f} "
              f"[{ci.loc[name, 0]:+.3f},{ci.loc[name, 1]:+.3f}]")
    print(f"  McFadden pseudo-R2 = {model.prsquared:.3f}; n = {int(model.nobs)}")

    # ---- 6. descriptive blocks ----------------------------------------------
    print("\n=== 6a. Mode B failure tags ===")
    for c in B_ARMS:
        tags = collections.Counter(r["failure_tag"] for r in b_by_arm[c])
        n = len(b_by_arm[c])
        print(f"  {c:12s} " + "  ".join(f"{k}={v / n:.3f}" for k, v in sorted(tags.items())))

    print("\n=== 6b. Mode A failure tags ===")
    for c in A_ARMS:
        tags = collections.Counter(r["failure_tag"] for r in a_by_arm[c])
        n = len(a_by_arm[c])
        print(f"  {c:4s} " + "  ".join(f"{k}={v / n:.3f}" for k, v in sorted(tags.items())))

    print("\n=== 6c. Mode A per-ability success ===")
    abilities = ["single-session-user", "single-session-assistant",
                 "knowledge-update", "multi-session", "temporal-reasoning",
                 "single-session-preference"]
    for a in abilities:
        row = [f"{a:28s}"]
        for c in A_ARMS:
            rs = [r for r in a_by_arm[c] if r["question_type"] == a]
            row.append(f"{c}={np.mean([r['success'] for r in rs]):.3f}(n={len(rs)})")
        print("  " + "  ".join(row))

    print("\n=== 6d. Cost, Mode B (metered totals) ===")
    base = np.mean([total_tokens(r) for r in b_by_arm["b"]])
    for c in B_ARMS:
        rs = b_by_arm[c]
        answ = np.mean([r["prompt_tokens"] for r in rs])
        tot = np.mean([total_tokens(r) for r in rs])
        calls = np.mean([r["total_llm_calls"] for r in rs])
        print(f"  {c:12s} answ={answ:.1f} total={tot:.1f} calls={calls:.2f} "
              f"answ_vs_b={100 * (answ - 1123.95) / 1123.95:+.1f}% "
              f"total_vs_b={100 * (tot - base) / base:+.1f}%")
    print("  per-seed m3 total vs b:")
    for s in (1, 2, 3):
        t = np.mean([total_tokens(r) for r in b_by_arm["m3"] if r["seed"] == s])
        tb = np.mean([total_tokens(r) for r in b_by_arm["b"] if r["seed"] == s])
        print(f"    seed{s}: {100 * (t - tb) / tb:+.1f}%")

    print("\n=== 6e. Cost, Mode A (instrumented records only) ===")
    base_a = None
    for c in A_ARMS:
        rs = [r for r in a_by_arm[c] if total_tokens(r) is not None]
        tot = np.mean([total_tokens(r) for r in rs])
        if c == "b":
            base_a = tot
        answ = np.mean([r["prompt_tokens"] for r in rs])
        calls = np.mean([r["total_llm_calls"] for r in rs])
        succ = np.mean([r["success"] for r in rs])
        print(f"  {c:4s} n={len(rs)} succ={succ:.3f} answ={answ:.0f} "
              f"total={tot:,.1f} calls={calls:.1f} x_base={tot / base_a:,.2f}")

    print("\n=== 6f. Seed spreads ===")
    for mode, by_arm, arms in (("B", b_by_arm, B_ARMS), ("A", a_by_arm, A_ARMS)):
        for c in arms:
            ss = [np.mean([r["success"] for r in by_arm[c] if r["seed"] == s])
                  for s in (1, 2, 3)]
            print(f"  Mode {mode} {c:12s} seeds={[round(x, 3) for x in ss]} "
                  f"spread={max(ss) - min(ss):.3f}")

    print("\n=== 6g. Baseline read coverage (Mode B) ===")
    rb = b_by_arm["b"]
    print(f"  n_retrieved mean={np.mean([r['n_retrieved'] for r in rb]):.1f}; "
          f"memory_items mean={np.mean([r['memory_items'] for r in rb]):.1f} "
          f"(minus current query turn = "
          f"{np.mean([r['memory_items'] for r in rb]) - 1:.1f})")


if __name__ == "__main__":
    main()
