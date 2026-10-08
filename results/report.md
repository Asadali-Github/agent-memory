> # ⚠ SUPERSEDED — generated from raw shards; authoritative numbers derive from `results/rerun_2026_07_29/*_merged.jsonl` via `scripts/ch4_stats.py` (Mode A 1,800 records, Mode B 1,971 records).

---

# Experiment report (auto-generated — do not hand-edit)

Generated: 2026-08-03T14:45:41.808740+00:00 | commit: 4852474

## Task: custom_ds

Records: 1290 | seeds: [1, 2, 3] | conditions: ['b', 'm1', 'm2', 'm3', 'm4']

### Headline summary
```
             n  success_rate  evidence_recall  prompt_tokens_answer_only  completion_tokens  memory_written_tokens  overhead_calls  latency_ms  success_ci_low  success_ci_high
condition                                                                                                                                                                      
b          258         0.760            0.802                   1124.012              6.655                1171.24           0.000     735.371           0.704            0.808
m1         258         0.795            1.000                   1125.891              5.655                1171.24           0.000     717.108           0.741            0.839
m2         258         0.593            0.802                   1124.012              7.376                1171.24           0.000     736.804           0.532            0.651
m3         258         0.733            0.996                    618.795              6.357                1171.24          21.566     957.685           0.675            0.783
m4         258         0.705            1.000                   1125.922              8.713                1171.24           0.000     775.021           0.647            0.758
```

### PRIMARY — success vs baseline: exact McNemar (Holm-corrected) + risk difference with bootstrap CI

_McNemar, not Wilcoxon: on 0/1 outcomes every non-zero difference has |d|=1 and the signed-rank statistic degenerates._

```
contrast  n_items  n_discordant  risk_diff  ci_low  ci_high  odds_ratio      p  p_holm  significant_05
 m1 vs b      258            97     0.0349 -0.0312   0.0988      1.2045 0.4168  0.8335           False
 m2 vs b      258            47    -0.1667 -0.2140  -0.1215      0.0444 0.0000  0.0000            True
 m3 vs b      258           101    -0.0271 -0.1059   0.0500      0.8704 0.5507  0.8335           False
 m4 vs b      258           116    -0.0543 -0.1236   0.0162      0.7846 0.2273  0.6818           False
```

_n_items is the MATCHED n. A paired test may only ever be quoted against this number, never against the raw per-condition record counts._

### Per-seed variance

```
           seed_1  seed_2  seed_3    mean  spread  n_seeds
condition                                                 
b          0.7381  0.7816  0.7586  0.7594  0.0435        3
m1         0.8095  0.7126  0.8621  0.7947  0.1494        3
m2         0.6190  0.5402  0.6207  0.5933  0.0805        3
m3         0.7738  0.7356  0.6897  0.7330  0.0842        3
m4         0.7500  0.6207  0.7471  0.7059  0.1293        3
```

**Effects smaller than their own seed-to-seed spread — do not report these as findings without more seeds:** m1 vs b (|effect|=0.035); m3 vs b (|effect|=0.027); m4 vs b (|effect|=0.054)

### RQ3 — COST contrasts (tokens, latency), Holm-corrected within metric

```
contrast            metric  mean_diff  pct_change    ci_low   ci_high  p_holm  significant_05
 m1 vs b     prompt_tokens     1.8798      0.1672    1.4264    2.3412  0.0000            True
 m2 vs b     prompt_tokens     0.0000      0.0000    0.0000    0.0000  1.0000           False
 m3 vs b     prompt_tokens  -505.2171    -44.9477 -516.7908 -493.4530  0.0000            True
 m4 vs b     prompt_tokens     1.9109      0.1700    1.4845    2.3488  0.0000            True
 m1 vs b completion_tokens    -1.0000    -15.0262   -1.4767   -0.5426  0.0000            True
 m2 vs b completion_tokens     0.7209     10.8328    0.2326    1.2558  0.0829           False
 m3 vs b completion_tokens    -0.2984     -4.4846   -0.8953    0.3101  0.3535           False
 m4 vs b completion_tokens     2.0581     30.9260   -0.4768    6.5740  0.3535           False
 m1 vs b        latency_ms   -18.2628     -2.4835  -40.5148   -1.6720  0.0000            True
 m2 vs b        latency_ms     1.4333      0.1949  -21.4729   19.3678  0.0018            True
 m3 vs b        latency_ms   222.3143     30.2316  171.6847  274.1682  0.0000            True
 m4 vs b        latency_ms    39.6500      5.3918  -14.1209  125.9378  0.0000            True
```

_WARNING: `prompt_tokens` counts the ANSWERING call only and excludes extraction/write calls, so it understates any condition with a write path. Re-run with the metered runner, or use `scripts/reconstruct_cost.py`, before quoting any token saving._

### Logistic regression (success ~ condition + evidence_distance, cluster-robust SEs)
```
                                  Results: Logit
==================================================================================
Model:                     Logit                 Method:                MLE       
Dependent Variable:        success               Pseudo R-squared:      0.031     
Date:                      2026-08-03 15:45      AIC:                   1500.8221 
No. Observations:          1290                  BIC:                   1531.7965 
Df Model:                  5                     Log-Likelihood:        -744.41   
Df Residuals:              1284                  LL-Null:               -768.47   
Converged:                 1.0000                LLR p-value:           3.3572e-09
No. Iterations:            5.0000                Scale:                 1.0000    
----------------------------------------------------------------------------------
                                    Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
----------------------------------------------------------------------------------
Intercept                           0.6418   0.1811  3.5441 0.0004  0.2869  0.9967
C(condition, Treatment('b'))[T.m1]  0.2044   0.1964  1.0409 0.2979 -0.1805  0.5894
C(condition, Treatment('b'))[T.m2] -0.7871   0.1169 -6.7356 0.0000 -1.0161 -0.5580
C(condition, Treatment('b'))[T.m3] -0.1454   0.2158 -0.6737 0.5005 -0.5684  0.2776
C(condition, Treatment('b'))[T.m4] -0.2818   0.1931 -1.4593 0.1445 -0.6604  0.0967
evidence_distance                   0.0084   0.0025  3.3276 0.0009  0.0035  0.0134
==================================================================================

```

#### Odds ratios with 95% CIs
```
                                      coef  odds_ratio  or_ci_low  or_ci_high       p
Intercept                           0.6418      1.8999     1.3322      2.7094  0.0004
C(condition, Treatment('b'))[T.m1]  0.2044      1.2268     0.8348      1.8028  0.2979
C(condition, Treatment('b'))[T.m2] -0.7871      0.4552     0.3620      0.5723  0.0000
C(condition, Treatment('b'))[T.m3] -0.1454      0.8647     0.5664      1.3200  0.5005
C(condition, Treatment('b'))[T.m4] -0.2818      0.7544     0.5166      1.1015  0.1445
evidence_distance                   0.0084      1.0084     1.0035      1.0134  0.0009
```

### Failure-mode composition (row-normalised)
```
failure_tag  reasoning_failure  retrieval_miss  stale_fact  success
condition                                                          
b                        0.000           0.198       0.043    0.760
m1                       0.000           0.000       0.205    0.795
m2                       0.000           0.190       0.217    0.593
m3                       0.132           0.000       0.136    0.733
m4                       0.031           0.000       0.264    0.705
```


---

# SENSITIVITY ANALYSIS — custom_ds, evidence_distance >= 20

Post-hoc filter applied in ANALYSIS, not enforced by the generator (true minimum distance in the full run is 1). Primary results are the full-sample section above (258 items); this subset (219 items, median distance 74) is reported to show the conclusions' robustness to the gap threshold. Effect sizes move materially between the two, so neither may be quoted without stating which sample it came from.

## Task: custom_ds (distance >= 20)

Records: 1095 | seeds: [1, 2, 3] | conditions: ['b', 'm1', 'm2', 'm3', 'm4']

### Headline summary
```
             n  success_rate  evidence_recall  prompt_tokens_answer_only  completion_tokens  memory_written_tokens  overhead_calls  latency_ms  success_ci_low  success_ci_high
condition                                                                                                                                                                      
b          219         0.744            0.767                   1123.950              6.895               1169.954           0.000     734.410           0.683            0.798
m1         219         0.836            1.000                   1125.689              5.717               1169.954           0.000     718.072           0.781            0.879
m2         219         0.626            0.767                   1123.950              7.735               1169.954           0.000     743.149           0.560            0.687
m3         219         0.758            1.000                    615.799              6.507               1169.954          21.594     970.424           0.697            0.810
m4         219         0.740            1.000                   1125.776              6.954               1169.954           0.000     743.173           0.678            0.793
```

### PRIMARY — success vs baseline: exact McNemar (Holm-corrected) + risk difference with bootstrap CI

_McNemar, not Wilcoxon: on 0/1 outcomes every non-zero difference has |d|=1 and the signed-rank statistic degenerates._

```
contrast  n_items  n_discordant  risk_diff  ci_low  ci_high  odds_ratio      p  p_holm  significant_05
 m1 vs b      219            82     0.0913  0.0179   0.1681      1.6452 0.0352  0.1057           False
 m2 vs b      219            30    -0.1187 -0.1690  -0.0736      0.0714 0.0000  0.0000            True
 m3 vs b      219            85     0.0137 -0.0660   0.0922      1.0732 0.8284  1.0000           False
 m4 vs b      219            97    -0.0046 -0.0837   0.0833      0.9796 1.0000  1.0000           False
```

_n_items is the MATCHED n. A paired test may only ever be quoted against this number, never against the raw per-condition record counts._

### Per-seed variance

```
           seed_1  seed_2  seed_3    mean  spread  n_seeds
condition                                                 
b          0.7297  0.7600  0.7429  0.7442  0.0303        3
m1         0.8514  0.7467  0.9143  0.8374  0.1676        3
m2         0.6486  0.5600  0.6714  0.6267  0.1114        3
m3         0.7838  0.7733  0.7143  0.7571  0.0695        3
m4         0.7838  0.6533  0.7857  0.7409  0.1324        3
```

**Effects smaller than their own seed-to-seed spread — do not report these as findings without more seeds:** m1 vs b (|effect|=0.091); m3 vs b (|effect|=0.014); m4 vs b (|effect|=0.005)

### RQ3 — COST contrasts (tokens, latency), Holm-corrected within metric

```
contrast            metric  mean_diff  pct_change    ci_low   ci_high  p_holm  significant_05
 m1 vs b     prompt_tokens     1.7397      0.1548    1.2557    2.2283  0.0000            True
 m2 vs b     prompt_tokens     0.0000      0.0000    0.0000    0.0000  1.0000           False
 m3 vs b     prompt_tokens  -508.1507    -45.2112 -520.2559 -495.5661  0.0000            True
 m4 vs b     prompt_tokens     1.8265      0.1625    1.3881    2.2695  0.0000            True
 m1 vs b completion_tokens    -1.1781    -17.0861   -1.7306   -0.6393  0.0000            True
 m2 vs b completion_tokens     0.8402     12.1854    0.2694    1.4749  0.0841           False
 m3 vs b completion_tokens    -0.3881     -5.6291   -1.0732    0.3242  0.2304           False
 m4 vs b completion_tokens     0.0594      0.8609   -0.7808    1.0366  0.2304           False
 m1 vs b        latency_ms   -16.3379     -2.2246  -36.4387   -0.9928  0.0003            True
 m2 vs b        latency_ms     8.7393      1.1900  -12.4268   25.1503  0.0027            True
 m3 vs b        latency_ms   236.0142     32.1366  182.2912  293.6523  0.0000            True
 m4 vs b        latency_ms     8.7630      1.1932  -14.8792   30.6179  0.0000            True
```

_WARNING: `prompt_tokens` counts the ANSWERING call only and excludes extraction/write calls, so it understates any condition with a write path. Re-run with the metered runner, or use `scripts/reconstruct_cost.py`, before quoting any token saving._

### Logistic regression (success ~ condition + evidence_distance, cluster-robust SEs)
```
                                  Results: Logit
==================================================================================
Model:                     Logit                 Method:                MLE       
Dependent Variable:        success               Pseudo R-squared:      0.022     
Date:                      2026-08-03 15:45      AIC:                   1237.3759 
No. Observations:          1095                  BIC:                   1267.3670 
Df Model:                  5                     Log-Likelihood:        -612.69   
Df Residuals:              1089                  LL-Null:               -626.76   
Converged:                 1.0000                LLR p-value:           3.3991e-05
No. Iterations:            6.0000                Scale:                 1.0000    
----------------------------------------------------------------------------------
                                    Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
----------------------------------------------------------------------------------
Intercept                           0.7440   0.2790  2.6664 0.0077  0.1971  1.2910
C(condition, Treatment('b'))[T.m1]  0.5587   0.2385  2.3422 0.0192  0.0912  1.0262
C(condition, Treatment('b'))[T.m2] -0.5566   0.1153 -4.8267 0.0000 -0.7826 -0.3306
C(condition, Treatment('b'))[T.m3]  0.0735   0.2182  0.3366 0.7364 -0.3543  0.5012
C(condition, Treatment('b'))[T.m4] -0.0239   0.2271 -0.1053 0.9161 -0.4690  0.4212
evidence_distance                   0.0045   0.0037  1.2123 0.2254 -0.0028  0.0118
==================================================================================

```

#### Odds ratios with 95% CIs
```
                                      coef  odds_ratio  or_ci_low  or_ci_high       p
Intercept                           0.7440      2.1044     1.2179      3.6363  0.0077
C(condition, Treatment('b'))[T.m1]  0.5587      1.7484     1.0955      2.7905  0.0192
C(condition, Treatment('b'))[T.m2] -0.5566      0.5732     0.4572      0.7185  0.0000
C(condition, Treatment('b'))[T.m3]  0.0735      1.0762     0.7017      1.6507  0.7364
C(condition, Treatment('b'))[T.m4] -0.0239      0.9764     0.6256      1.5238  0.9161
evidence_distance                   0.0045      1.0045     0.9972      1.0119  0.2254
```

### Failure-mode composition (row-normalised)
```
failure_tag  reasoning_failure  retrieval_miss  stale_fact  success
condition                                                          
b                        0.000           0.233       0.023    0.744
m1                       0.000           0.000       0.164    0.836
m2                       0.000           0.224       0.151    0.626
m3                       0.137           0.000       0.105    0.758
m4                       0.032           0.000       0.228    0.740
```

## Task: longmemeval_s

Records: 461 | seeds: [1, 2, 3] | conditions: ['b', 'm1']

### Headline summary
```
             n  success_rate  evidence_recall  prompt_tokens_answer_only  completion_tokens  memory_written_tokens  overhead_calls  latency_ms  success_ci_low  success_ci_high
condition                                                                                                                                                                      
b          341         0.032            0.029                   1252.607             32.422             111725.598             0.0    1325.433           0.018            0.057
m1         120         0.433            0.728                   1250.425             28.867             111780.108             0.0    2184.458           0.348            0.523
```

### PRIMARY — success vs baseline: exact McNemar (Holm-corrected) + risk difference with bootstrap CI

_McNemar, not Wilcoxon: on 0/1 outcomes every non-zero difference has |d|=1 and the signed-rank statistic degenerates._

```
contrast  n_items  n_discordant  risk_diff  ci_low  ci_high  odds_ratio   p  p_holm  significant_05
 m1 vs b      120            51     0.4083  0.3167      0.5        50.0 0.0     0.0            True
```

_n_items is the MATCHED n. A paired test may only ever be quoted against this number, never against the raw per-condition record counts._

### Per-seed variance

```
           seed_1  seed_2  seed_3    mean  spread  n_seeds
condition                                                 
b          0.0504  0.0250  0.0196  0.0317  0.0308        3
m1            NaN  0.4333     NaN  0.4333  0.0000        1
```

### RQ3 — COST contrasts (tokens, latency), Holm-corrected within metric

```
contrast            metric  mean_diff  pct_change   ci_low   ci_high  p_holm  significant_05
 m1 vs b     prompt_tokens    -1.9417     -0.1550  -3.4917   -0.4083  0.0185            True
 m1 vs b completion_tokens    -2.9250     -9.0216 -12.3335    5.0335  0.0494            True
 m1 vs b        latency_ms   869.1193     65.5725 699.8784 1013.7232  0.0000            True
```

_WARNING: `prompt_tokens` counts the ANSWERING call only and excludes extraction/write calls, so it understates any condition with a write path. Re-run with the metered runner, or use `scripts/reconstruct_cost.py`, before quoting any token saving._

### Logistic regression
_Not estimable on this data: Singular matrix_

### Failure-mode composition (row-normalised)
```
failure_tag  partial_retrieval  reasoning_failure  retrieval_miss  success
condition                                                                 
b                        0.006               0.05           0.912    0.032
m1                       0.092               0.30           0.175    0.433
```
