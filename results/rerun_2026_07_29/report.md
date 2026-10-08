> # ⚠ SUPERSEDED — generated from raw shards; authoritative numbers derive from `results/rerun_2026_07_29/*_merged.jsonl` via `scripts/ch4_stats.py` (Mode A 1,800 records, Mode B 1,971 records).

---

# Experiment report (auto-generated — do not hand-edit)

Generated: 2026-08-02T20:35:10.753868+00:00 | commit: unknown

## Task: custom_ds

Records: 1971 | seeds: [1, 2, 3] | conditions: ['b', 'b_reversed', 'm1', 'm1_chrono', 'm2', 'm3', 'm3_chrono', 'm4', 'm4_chrono']

### Headline summary
```
              n  success_rate  evidence_recall  prompt_tokens_answer_only  completion_tokens  memory_written_tokens  overhead_calls  latency_ms  total_prompt_tokens  total_llm_calls  success_ci_low  success_ci_high
condition                                                                                                                                                                                                             
b           219         0.744            0.767                   1123.950              6.895               1169.954           0.000     748.088             1123.950            1.000           0.683            0.798
b_reversed  219         0.626            0.767                   1123.950              7.735               1169.954           0.000     733.160             1123.950            1.000           0.560            0.687
m1          219         0.836            1.000                   1125.689              5.717               1169.954           0.000     724.425             1125.689            1.000           0.781            0.879
m1_chrono   219         0.973            1.000                   1125.689              5.667               1169.954           0.000     707.193             1125.689            1.000           0.942            0.987
m2          219         0.913            1.000                    991.365              4.863               1169.954           1.397     625.026             1210.954            1.274           0.868            0.944
m3          219         0.753            1.000                    616.721              6.306               1169.954          21.256     951.126             1049.982            4.584           0.692            0.806
m3_chrono   219         0.740            1.000                    616.721              5.950               1169.954          21.256     957.710             1049.982            4.584           0.678            0.793
m4          219         0.740            1.000                   1125.776              6.954               1169.954           0.000     735.865             1125.776            1.000           0.678            0.793
m4_chrono   219         0.973            1.000                   1125.776              5.840               1169.954           0.000     710.809             1125.776            1.000           0.942            0.987
```

### PRIMARY — success vs baseline: exact McNemar (Holm-corrected) + risk difference with bootstrap CI

_McNemar, not Wilcoxon: on 0/1 outcomes every non-zero difference has |d|=1 and the signed-rank statistic degenerates._

```
       contrast  n_items  n_discordant  risk_diff  ci_low  ci_high  odds_ratio      p  p_holm  significant_05
        m1 vs b      219            82     0.0913  0.0179   0.1681      1.6452 0.0352  0.1057           False
        m2 vs b      219            67     0.1689  0.0972   0.2427      3.4667 0.0000  0.0000            True
        m3 vs b      219            88     0.0091 -0.0702   0.0880      1.0465 0.9152  1.0000           False
        m4 vs b      219            97    -0.0046 -0.0837   0.0833      0.9796 1.0000  1.0000           False
b_reversed vs b      219            30    -0.1187 -0.1690  -0.0736      0.0714 0.0000  0.0000            True
```

_n_items is the MATCHED n. A paired test may only ever be quoted against this number, never against the raw per-condition record counts._

### Per-seed variance

```
            seed_1  seed_2  seed_3    mean  spread  n_seeds
condition                                                  
b           0.7297  0.7600  0.7429  0.7442  0.0303        3
b_reversed  0.6486  0.5600  0.6714  0.6267  0.1114        3
m1          0.8514  0.7467  0.9143  0.8374  0.1676        3
m1_chrono   0.9459  0.9867  0.9857  0.9728  0.0407        3
m2          0.9189  0.8800  0.9429  0.9139  0.0629        3
m3          0.7568  0.7867  0.7143  0.7526  0.0724        3
m3_chrono   0.7297  0.7733  0.7143  0.7391  0.0590        3
m4          0.7838  0.6533  0.7857  0.7409  0.1324        3
m4_chrono   0.9459  0.9867  0.9857  0.9728  0.0407        3
```

**Effects smaller than their own seed-to-seed spread — do not report these as findings without more seeds:** m1 vs b (|effect|=0.091); m3 vs b (|effect|=0.009); m4 vs b (|effect|=0.005)

### RQ3 — COST contrasts (tokens, latency), Holm-corrected within metric

```
       contrast                  metric  mean_diff  pct_change    ci_low  ci_high  p_holm  significant_05
        m1 vs b     total_prompt_tokens     1.7397      0.1548    1.2557   2.2283  0.0000            True
        m2 vs b     total_prompt_tokens    87.0046      7.7410   42.0218 132.8321  0.7936           False
        m3 vs b     total_prompt_tokens   -73.9680     -6.5811 -164.3243  19.2360  0.0000            True
        m4 vs b     total_prompt_tokens     1.8265      0.1625    1.3881   2.2695  0.0000            True
b_reversed vs b     total_prompt_tokens     0.0000      0.0000    0.0000   0.0000  1.0000           False
        m1 vs b total_completion_tokens    -1.1781    -17.0861   -1.7306  -0.6393  0.0000            True
        m2 vs b total_completion_tokens    93.2374   1352.2517   72.0495 114.8362  0.0010            True
        m3 vs b total_completion_tokens   135.6941   1968.0132  104.2048 168.6532  0.0000            True
        m4 vs b total_completion_tokens     0.0594      0.8609   -0.7808   1.0366  0.1620           False
b_reversed vs b total_completion_tokens     0.8402     12.1854    0.2694   1.4749  0.0561           False
        m1 vs b              latency_ms   -23.6630     -3.1631  -65.8529   2.0344  0.0001            True
        m2 vs b              latency_ms  -123.0626    -16.4503 -191.7393 -82.5391  0.0000            True
        m3 vs b              latency_ms   203.0384     27.1410  115.7403 273.5254  0.0000            True
        m4 vs b              latency_ms   -12.2233     -1.6339  -83.1466  31.9297  0.0000            True
b_reversed vs b              latency_ms   -14.9279     -1.9955  -83.5142  24.2507  0.0277            True
```

### Logistic regression (success ~ condition + evidence_distance, cluster-robust SEs)
```
                                      Results: Logit
==========================================================================================
Model:                       Logit                    Method:                   MLE       
Dependent Variable:          success                  Pseudo R-squared:         0.110     
Date:                        2026-08-02 21:35         AIC:                      1721.3971 
No. Observations:            1971                     BIC:                      1777.2601 
Df Model:                    9                        Log-Likelihood:           -850.70   
Df Residuals:                1961                     LL-Null:                  -956.18   
Converged:                   1.0000                   LLR p-value:              1.6574e-40
No. Iterations:              8.0000                   Scale:                    1.0000    
------------------------------------------------------------------------------------------
                                            Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
------------------------------------------------------------------------------------------
Intercept                                   0.3095   0.2947  1.0503 0.2936 -0.2681  0.8871
C(condition, Treatment('b'))[T.b_reversed] -0.5634   0.1173 -4.8027 0.0000 -0.7934 -0.3335
C(condition, Treatment('b'))[T.m1]          0.5641   0.2401  2.3494 0.0188  0.0935  1.0346
C(condition, Treatment('b'))[T.m1_chrono]   2.5176   0.4348  5.7910 0.0000  1.6655  3.3698
C(condition, Treatment('b'))[T.m2]          1.2976   0.3128  4.1484 0.0000  0.6845  1.9107
C(condition, Treatment('b'))[T.m3]          0.0492   0.2211  0.2226 0.8239 -0.3841  0.4826
C(condition, Treatment('b'))[T.m3_chrono]  -0.0242   0.2298 -0.1052 0.9162 -0.4745  0.4261
C(condition, Treatment('b'))[T.m4]         -0.0242   0.2296 -0.1053 0.9162 -0.4743  0.4259
C(condition, Treatment('b'))[T.m4_chrono]   2.5176   0.4348  5.7910 0.0000  1.6655  3.3698
evidence_distance                           0.0107   0.0037  2.8941 0.0038  0.0035  0.0179
==========================================================================================

```

#### Odds ratios with 95% CIs
```
                                              coef  odds_ratio  or_ci_low  or_ci_high       p
Intercept                                   0.3095      1.3628     0.7649      2.4280  0.2936
C(condition, Treatment('b'))[T.b_reversed] -0.5634      0.5693     0.4523      0.7164  0.0000
C(condition, Treatment('b'))[T.m1]          0.5641      1.7578     1.0980      2.8141  0.0188
C(condition, Treatment('b'))[T.m1_chrono]   2.5176     12.3994     5.2886     29.0713  0.0000
C(condition, Treatment('b'))[T.m2]          1.2976      3.6606     1.9829      6.7578  0.0000
C(condition, Treatment('b'))[T.m3]          0.0492      1.0504     0.6810      1.6202  0.8239
C(condition, Treatment('b'))[T.m3_chrono]  -0.0242      0.9761     0.6222      1.5313  0.9162
C(condition, Treatment('b'))[T.m4]         -0.0242      0.9761     0.6223      1.5310  0.9162
C(condition, Treatment('b'))[T.m4_chrono]   2.5176     12.3994     5.2886     29.0713  0.0000
evidence_distance                           0.0107      1.0108     1.0035      1.0181  0.0038
```

### Failure-mode composition (row-normalised)
```
failure_tag  reasoning_failure  retrieval_miss  stale_fact  success
condition                                                          
b                        0.000           0.233       0.023    0.744
b_reversed               0.000           0.224       0.151    0.626
m1                       0.000           0.000       0.164    0.836
m1_chrono                0.000           0.000       0.027    0.973
m2                       0.032           0.000       0.055    0.913
m3                       0.132           0.000       0.114    0.753
m3_chrono                0.137           0.000       0.123    0.740
m4                       0.032           0.000       0.228    0.740
m4_chrono                0.000           0.000       0.027    0.973
```

## Task: longmemeval_s

Records: 1799 | seeds: [1, 2, 3] | conditions: ['b', 'm1', 'm2', 'm3', 'm4']

### Headline summary
```
             n  success_rate  evidence_recall  prompt_tokens_answer_only  completion_tokens  memory_written_tokens  overhead_calls  latency_ms  total_prompt_tokens  total_llm_calls  success_ci_low  success_ci_high
condition                                                                                                                                                                                                            
b          359         0.039            0.028                   1252.588             31.788             111726.582           0.000    1311.169             1252.742            1.000           0.023            0.064
m1         360         0.408            0.749                   1251.122             27.878             111725.306           0.000    2176.091             1251.471            1.000           0.359            0.460
m2         360         0.042            1.000                   1025.303             25.050             111725.306         239.181    1036.583           220507.339          240.181           0.025            0.068
m3         360         0.217            0.741                   1195.525             16.022             111725.306         184.522    6368.942           122805.019          185.522           0.177            0.262
m4         360         0.236            0.271                   1250.831             24.761             111725.306           0.000    3014.912             1250.831            1.000           0.195            0.283
```

### PRIMARY — success vs baseline: exact McNemar (Holm-corrected) + risk difference with bootstrap CI

_McNemar, not Wilcoxon: on 0/1 outcomes every non-zero difference has |d|=1 and the signed-rank statistic degenerates._

```
contrast  n_items  n_discordant  risk_diff  ci_low  ci_high  odds_ratio   p  p_holm  significant_05
 m1 vs b      262           109     0.3855  0.3244   0.4466     26.2500 0.0     0.0            True
 m2 vs b      262             4     0.0038 -0.0115   0.0210      1.0000 1.0     1.0           False
 m3 vs b      262            55     0.1870  0.1374   0.2366     17.3333 0.0     0.0            True
 m4 vs b      262            51     0.1947  0.1489   0.2443         inf 0.0     0.0            True
```

_n_items is the MATCHED n. A paired test may only ever be quoted against this number, never against the raw per-condition record counts._

### Per-seed variance

```
           seed_1  seed_2  seed_3    mean  spread  n_seeds
condition                                                 
b          0.0504  0.0250  0.0417  0.0390  0.0254        3
m1         0.3917  0.4333  0.4000  0.4083  0.0417        3
m2         0.0500  0.0333  0.0417  0.0417  0.0167        3
m3         0.2083  0.2083  0.2333  0.2167  0.0250        3
m4         0.2333  0.2333  0.2417  0.2361  0.0083        3
```

**Effects smaller than their own seed-to-seed spread — do not report these as findings without more seeds:** m2 vs b (|effect|=0.004)

### RQ3 — COST contrasts (tokens, latency), Holm-corrected within metric

```
contrast                  metric   mean_diff  pct_change      ci_low     ci_high  p_holm  significant_05
 m1 vs b     total_prompt_tokens     -0.6750     -0.0539     -2.2000      0.8333  0.4795           False
 m2 vs b     total_prompt_tokens 218697.6042  17457.5182 217897.4031 219495.4830  0.0000            True
 m3 vs b     total_prompt_tokens 121282.8083   9681.3902 120906.4383 121634.4977  0.0000            True
 m4 vs b     total_prompt_tokens     -1.6417     -0.1310     -3.3502      0.0583  0.1259           False
 m1 vs b total_completion_tokens     -4.3250    -13.9968    -13.4083      3.5917  0.0313            True
 m2 vs b total_completion_tokens  95307.2333 308437.6483  94800.3419  95840.1688  0.0000            True
 m3 vs b total_completion_tokens  40838.8250 132164.4822  40475.4983  41208.1504  0.0000            True
 m4 vs b total_completion_tokens     -7.1000    -22.9773    -16.6833      0.0417  0.0046            True
 m1 vs b              latency_ms    862.1330     65.7530    775.2156    939.0596  0.0000            True
 m2 vs b              latency_ms   -223.1007    -17.0154   -310.5499   -154.4582  0.0000            True
 m3 vs b              latency_ms   5216.2795    397.8342   4799.2133   5652.5764  0.0000            True
 m4 vs b              latency_ms   1817.5144    138.6178   1507.0164   2251.3535  0.0000            True
```

### Logistic regression (success ~ condition + question_type, cluster-robust SEs)
```
                                         Results: Logit
=================================================================================================
Model:                          Logit                      Method:                     MLE       
Dependent Variable:             success                    Pseudo R-squared:           0.253     
Date:                           2026-08-02 21:35           AIC:                        1321.2802 
No. Observations:               1799                       BIC:                        1376.2301 
Df Model:                       9                          Log-Likelihood:             -650.64   
Df Residuals:                   1789                       LL-Null:                    -870.63   
Converged:                      0.0000                     LLR p-value:                3.9892e-89
No. Iterations:                 35.0000                    Scale:                      1.0000    
-------------------------------------------------------------------------------------------------
                                               Coef.   Std.Err.    z     P>|z|   [0.025   0.975] 
-------------------------------------------------------------------------------------------------
Intercept                                      -2.8865   0.3885  -7.4304 0.0000  -3.6479  -2.1251
C(condition, Treatment('b'))[T.m1]              3.0651   0.4075   7.5222 0.0000   2.2665   3.8637
C(condition, Treatment('b'))[T.m2]              0.0704   0.2223   0.3168 0.7514  -0.3653   0.5062
C(condition, Treatment('b'))[T.m3]              2.0076   0.3981   5.0431 0.0000   1.2274   2.7878
C(condition, Treatment('b'))[T.m4]              2.1303   0.3528   6.0384 0.0000   1.4389   2.8218
C(question_type)[T.multi-session]              -1.1268   0.2941  -3.8311 0.0001  -1.7032  -0.5503
C(question_type)[T.single-session-assistant]    0.1677   0.2742   0.6117 0.5407  -0.3697   0.7052
C(question_type)[T.single-session-preference] -20.6558   0.2726 -75.7634 0.0000 -21.1902 -20.1215
C(question_type)[T.single-session-user]         0.2490   0.2597   0.9586 0.3378  -0.2601   0.7580
C(question_type)[T.temporal-reasoning]         -0.4390   0.4360  -1.0069 0.3140  -1.2936   0.4155
=================================================================================================

```

#### Odds ratios with 95% CIs
```
                                                  coef  odds_ratio  or_ci_low  or_ci_high       p
Intercept                                      -2.8865      0.0558     0.0260      0.1194  0.0000
C(condition, Treatment('b'))[T.m1]              3.0651     21.4363     9.6452     47.6419  0.0000
C(condition, Treatment('b'))[T.m2]              0.0704      1.0730     0.6940      1.6590  0.7514
C(condition, Treatment('b'))[T.m3]              2.0076      7.4454     3.4122     16.2456  0.0000
C(condition, Treatment('b'))[T.m4]              2.1303      8.4177     4.2159     16.8072  0.0000
C(question_type)[T.multi-session]              -1.1268      0.3241     0.1821      0.5768  0.0001
C(question_type)[T.single-session-assistant]    0.1677      1.1826     0.6909      2.0243  0.5407
C(question_type)[T.single-session-preference] -20.6558      0.0000     0.0000      0.0000  0.0000
C(question_type)[T.single-session-user]         0.2490      1.2827     0.7710      2.1340  0.3378
C(question_type)[T.temporal-reasoning]         -0.4390      0.6447     0.2743      1.5152  0.3140
```

### Failure-mode composition (row-normalised)
```
failure_tag  partial_retrieval  reasoning_failure  retrieval_miss  success
condition                                                                 
b                        0.006              0.047           0.908    0.039
m1                       0.103              0.328           0.161    0.408
m2                       0.000              0.958           0.000    0.042
m3                       0.153              0.483           0.147    0.217
m4                       0.142              0.089           0.533    0.236
```
