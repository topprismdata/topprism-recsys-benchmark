# GPT Final Review Verdict — TopPrism RecSys Benchmark v0.9 (R0–R3)

- **Date:** 2026-10-26
- **Channel:** ChatGPT formal session `6adc27a7-7784-83ee-a6a1-6ef93d31d2a2` (browser relay)
- **Full reply archive:** `.superpowers/sdd/2026-10-25-topprism-recsys-benchmark-r0-r3-plan/gpt-final-review-reply-2026-10-26.txt` (local ledger, gitignored)
- **Evidence index:** branch `feat/recsys-benchmark-v09`, commits `8a736bd → f892b04`; `pytest` 27 passed; Task4 independent reviewer PASS.

## Overall verdict

**APPROVE WITH CONDITIONS** — research implementation accepted; REPORT DRAFT phase approved now; FINAL RESEARCH VERDICT requires five targeted confirmations (listed below). No unconditional mathematical-attribution claims or technology-route promotions approved yet.

## Per-item verdicts

| Item | Verdict |
|---|---|
| R0 split/evaluator/ExperimentIR/provenance/seal | PASS; original HOLD lifted (Jaccard date pairs, SHA-256 fingerprints, 240,311 excluded W5 rows form a consistent seal chain) |
| R1 W4 results | PASS; W2–W4 walk-forward accepted with documentation condition (per-window out-of-time invariants must be proven at final archive) |
| R2 orthogonal matrix | PASS as experimental evidence; attribution PARTIAL |
| R2 H&M BCE-256 arm as final architecture ranking | REJECT (under-training demonstrated by epochs 8→24 probe, HSTU +0.0175) |
| R2 SS256 arm within-budget relative comparison | CONDITIONAL PASS |
| R2 shuffle sensitivity (descriptive cross-domain difference) | PASS |
| R2 "Attention provides independent predictive gain on H&M" | NOT ESTABLISHED (order-sensitive but not beating MeanPool/GBDT) |
| R2 strong causal double-dissociation of temporal information value | INCONCLUSIVE |
| R3 performance measurement / AMP / SDPA engineering gain | PASS (SDPA limited to tested config) |
| R3 numerical exact equivalence | NOT ESTABLISHED (top-12 agreement 0.981–0.999 must be reported as output-changing) |
| R3 compute-bottleneck root cause | INCONCLUSIVE |
| C1 domain-conditioned information gain | PARTIAL PASS (repeat-domain GBDT advantage PASS; cross-domain order-sensitivity difference PASS descriptive) |
| C2 protocol-effect separability | METHOD PASS, effect PARTIAL/INCONCLUSIVE |
| C3 Hybrid gain tracks repeat share | swire local PASS (+0.0017, CI [+0.0003,+0.0029], p=0.014, multiplicity disclosure required); cross-domain causal explanation HOLD |
| C4 single-GPU compute frontier | PASS on engineering; pure-framework equivalence conditional |

## Required confirmations before FINAL RESEARCH VERDICT

1. R1 per-window out-of-time invariant acceptance records + explanation of R1 vs R2 baseline numeric differences (B1 0.4246→0.4255, B2 0.4319→0.4299/0.4300).
2. H&M enlarged pre-frozen confirmation sample with paired CIs (esp. S2: GRU/Attention vs GBDT and vs MeanPool).
3. GRU/SASRec confirmation experiment: true-order vs shuffled-inference vs shuffled-training vs recency-preserving order-free control, multiple shuffle seeds, unified pool/budget.
4. HSTU-Math key mathematical consistency evidence vs reference implementation + convergence diagnostics for under-trained models.
5. Hybrid same-protocol B2-vs-B2+Neural paired comparison on identical candidates/customers/windows + R3 joint report of numeric error vs recommendation accuracy vs throughput.

## Allowed / forbidden report statements (excerpt)

Allowed: unified-protocol empirical comparison across two structurally different retail domains; swire GBDT lead + small embedding gain; H&M low exact-SKU repeat but high product-type preference persistence; H&M order sensitivity concentrated in S2 but not yet converted into overall advantage over strong traditional baselines; AMP/SDPA observable efficiency gains implementation- and precision-dependent.

Forbidden: "proved Attention math beats traditional math in exploratory retail"; "proved HSTU unsuitable for recommender systems"; "proved repeat share causally determines Hybrid gain"; "proved 2022 failure was GPU architecture"; "proved more model/compute cannot help".
