# Final consistency audit 2026-08-04 — verify manuscript claims not already
# covered byte-for-byte by ch4_stats_output_2026-08-04.txt.
import json, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = [json.loads(l) for l in open(ROOT / "rerun_2026_07_29/custom_ds_merged.jsonl")]
A = [json.loads(l) for l in open(ROOT / "rerun_2026_07_29/longmemeval_s_merged.jsonl")]
print(f"Mode B records: {len(B)}  Mode A records: {len(A)}")

def by_cond(recs):
    d = defaultdict(list)
    for r in recs:
        d[r["condition"]].append(r)
    return d

bc, ac = by_cond(B), by_cond(A)

print("\n=== Mode B: per-arm n and per-seed counts ===")
for c in sorted(bc):
    seeds = defaultdict(int)
    for r in bc[c]:
        seeds[r["seed"]] += 1
    print(f"  {c:12s} n={len(bc[c])}  seeds={dict(sorted(seeds.items()))}")

print("\n=== Ch3 DP5: evidence distance stats (unique items via arm b) ===")
dists = [r["evidence_distance"] for r in bc["b"]]
print(f"  min={min(dists)} median={st.median(dists)} max={max(dists)}  (claim: min 20, median 74, max 117)")
# check identical across arms
for c in sorted(bc):
    d2 = sorted(r["evidence_distance"] for r in bc[c])
    if d2 != sorted(dists):
        print(f"  WARNING: {c} distances differ from b")

print("\n=== Ch3 M2 threshold sweep: history tokens (baseline memory_written_tokens) ===")
ht = [r["memory_written_tokens"] for r in bc["b"]]
print(f"  min={min(ht)} median={st.median(ht)} mean={st.mean(ht):.1f} max={max(ht)}")
print(f"  (claim: min 872, median 1168, mean 1170, max 1453)")
for thr in (1500, 1200, 1000, 800):
    n = sum(1 for t in ht if t > thr)
    print(f"  thr {thr}: fires on {n}/219  (claims: 0, 92, 185, 219)")
print(f"  92/219={92/219:.3f} (claim 42%), 185/219={185/219:.3f} (claim 84%)")

print("\n=== Ch4 depth text: baseline RECALL at d>=80 (claim in text: 0.452) ===")
for thr in (20, 40, 60, 80):
    sub = [r for r in bc["b"] if r["evidence_distance"] >= thr]
    rec = st.mean(r["evidence_recall"] for r in sub)
    suc = st.mean(1.0 if r["success"] else 0.0 for r in sub)
    print(f"  d>={thr}: n={len(sub)} b recall={rec:.3f} success={suc:.3f}")

print("\n=== Ch4: M2 per-seed TOTAL tokens vs b (claim: +13.8% to +17.5%) ===")
for s in (1, 2, 3):
    tb = st.mean(r["total_prompt_tokens"] + r["total_completion_tokens"] for r in bc["b"] if r["seed"] == s)
    tm = st.mean(r["total_prompt_tokens"] + r["total_completion_tokens"] for r in bc["m2"] if r["seed"] == s)
    print(f"  seed{s}: b={tb:.1f} m2={tm:.1f}  {(tm/tb-1)*100:+.1f}%")
print("  M3 per-seed (claim +2.2/+8.4/+5.8):")
for s in (1, 2, 3):
    tb = st.mean(r["total_prompt_tokens"] + r["total_completion_tokens"] for r in bc["b"] if r["seed"] == s)
    tm = st.mean(r["total_prompt_tokens"] + r["total_completion_tokens"] for r in bc["m3"] if r["seed"] == s)
    print(f"  seed{s}: {(tm/tb-1)*100:+.1f}%")

print("\n=== Ch4: baseline reads 88.9 of 95.5 ===")
nr = st.mean(r["n_retrieved"] for r in bc["b"])
mi = st.mean(r["memory_items"] for r in bc["b"])
print(f"  n_retrieved mean={nr:.1f}  memory_items mean={mi:.1f} (minus current turn = {mi-1:.1f})")

print("\n=== Mode B queries per episode / placement (item_id epX_YYY:qZZZ) ===")
import re
qs = defaultdict(int)
qsteps = []
for r in bc["b"]:
    m = re.match(r"ep(\d+)_(\d+):q(\d+)", r["item_id"])
    if m:
        qs[(r["seed"], int(m.group(1)))] += 1
        qsteps.append(int(m.group(3)))
print(f"  episodes per seed: {len(set(k[1] for k in qs))}, queries/episode min={min(qs.values())} max={max(qs.values())}")
print(f"  query step index min={min(qsteps)} max={max(qsteps)} (final 40% of 120 steps = steps >=72)")

print("\n=== Mode A: abstention counts per condition ===")
for c in sorted(ac):
    absn = sum(1 for r in ac[c] if r["item_id"].endswith("_abs"))
    nullrec = sum(1 for r in ac[c] if r["evidence_recall"] is None)
    print(f"  {c}: _abs items={absn}, null evidence_recall={nullrec}  (claims: 12 and 10)")

print("\n=== Mode A: instrumented record counts (total_prompt_tokens present) ===")
for c in sorted(ac):
    inst = [r for r in ac[c] if r.get("total_prompt_tokens") is not None]
    suc = st.mean(1.0 if r["success"] else 0.0 for r in inst)
    print(f"  {c}: n={len(inst)} succ={suc:.3f}  (claims: b 121/0.041, m1 240/0.396, others 360)")

print("\n=== Mode A cost table recompute (instrumented only) ===")
for c in ("b", "m1", "m2", "m3", "m4"):
    inst = [r for r in ac[c] if r.get("total_prompt_tokens") is not None]
    tot = st.mean(r["total_prompt_tokens"] + r["total_completion_tokens"] for r in inst)
    calls = st.mean(r["total_llm_calls"] for r in inst)
    answ = st.mean(r["prompt_tokens"] for r in inst)
    print(f"  {c}: answ={answ:.0f} total={tot:,.1f} calls={calls:.1f}")
b_tot = st.mean(r["total_prompt_tokens"] + r["total_completion_tokens"] for r in ac["b"] if r.get("total_prompt_tokens") is not None)
for c in ("m1", "m2", "m3", "m4"):
    inst = [r for r in ac[c] if r.get("total_prompt_tokens") is not None]
    tot = st.mean(r["total_prompt_tokens"] + r["total_completion_tokens"] for r in inst)
    print(f"  {c} x_base = {tot/b_tot:.2f}")

print("\n=== Mode A per-ability success ===")
qa = defaultdict(lambda: defaultdict(list))
for r in A:
    qa[r["question_type"]][r["condition"]].append(1.0 if r["success"] else 0.0)
for qt in sorted(qa):
    row = "  " + qt + ": " + "  ".join(f"{c}={st.mean(qa[qt][c]):.3f}(n={len(qa[qt][c])})" for c in ("b", "m1", "m2", "m3", "m4"))
    print(row)

print("\n=== Mode A ability items per type per seed (claim: 20 each) ===")
cnt = defaultdict(int)
for r in ac["b"]:
    cnt[(r["seed"], r["question_type"])] += 1
for k in sorted(cnt):
    print(f"  seed{k[0]} {k[1]}: {cnt[k]}")

print("\n=== Pre-fix degenerate M2 (threshold 1500) vs b_reversed on 219 panel ===")
pf_path = ROOT / "runs/custom_ds.jsonl"
if pf_path.exists():
    PF = [json.loads(l) for l in open(pf_path)]
    panel = {(r["item_id"], r["seed"]) for r in bc["b"]}
    brev = {(r["item_id"], r["seed"]): r["success"] for r in bc["b_reversed"]}
    pfm2 = [r for r in PF if r["condition"] == "m2" and (r["item_id"], r["seed"]) in panel]
    print(f"  pre-fix m2 records on panel: {len(pfm2)}")
    if pfm2:
        suc = st.mean(1.0 if r["success"] else 0.0 for r in pfm2)
        agree = sum(1 for r in pfm2 if r["success"] == brev[(r["item_id"], r["seed"])])
        print(f"  pre-fix m2 success on panel = {suc:.4f} (claim 0.6256); agrees with b_reversed on {agree}/219")
else:
    print("  pre-fix file not found:", pf_path)

print("\n=== Reproducibility check: pre-fix b/m1/m4 vs merged on 219 panel ===")
if pf_path.exists():
    for c in ("b", "m1", "m4"):
        pf = {(r["item_id"], r["seed"]): r["success"] for r in PF if r["condition"] == c and (r["item_id"], r["seed"]) in panel}
        cur = {(r["item_id"], r["seed"]): r["success"] for r in bc[c]}
        if pf:
            agree = sum(1 for k in pf if pf[k] == cur[k])
            ps = st.mean(1.0 if v else 0.0 for v in pf.values())
            print(f"  {c}: pre-fix n={len(pf)} succ={ps:.3f}, agree {agree}/{len(pf)}")
