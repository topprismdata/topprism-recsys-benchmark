# TopPrism RecSys Benchmark v0.9 — Research Report DRAFT

- **Status:** FINAL-READY (GPT FINAL RESEARCH VERDICT 2026-10-26: **APPROVED WITH SCOPE LIMITATIONS**; approved to proceed to final editing and archival. Three items remain explicitly PENDING — see §8. Verdict archive: `.superpowers/sdd/.../gpt-final-verdict-2026-10-26.txt`.)
- **Evidence index:** branch `feat/recsys-benchmark-v09`, commits `8a736bd → fe36ca2`; frozen contract `protocols/recsys-contract-v0.9.json`; `pytest` 28 passed (12 R0 golden + R1 pipeline + R2 orthogonal + order-mode unit tests); independent code reviews PASS (Tasks 1–4); GPT final review `docs/superpowers/reviews/2026-10-26-gpt-final-review.md`.
- **Domains:** `swire_ro` (B2B fast-moving goods replenishment, ~88.3% exact-SKU repeat) and `H&M` (Kaggle fashion transactions, 3.93% exact-SKU repeat, 8.28% product_code style-family repeat, 61.98% product_type_no history-hit).

## 1. Research questions

- **RQ1 (information):** Do behavioural representations (embeddings, behavioural sequences) add predictive information beyond explicit statistical features?
- **RQ2 (mechanism):** Which mathematical mechanism (order-free pooling vs recurrent vs self-attention vs HSTU-style attention math; loss/negative-sampling protocol) carries any gain?
- **RQ3 (engineering):** What does the accuracy–cost frontier look like on a single consumer GPU (RTX 3080Ti 20GB)?

Evaluation is MAP@12 on a 7-day horizon with Kaggle-compatible scored-customer denominators, deterministic tie-breaking, and sealed W5 labels (never touched; 240,311 post-cutoff H&M rows excluded; SHA-256 fingerprints in `artifacts/r0/hm_r0_multi_granularity_addendum.json`).

## 2. R0 — data contract and dual-domain profiles

- Unified canonical event table (customer_id, article_id, event_date, value); W1..W4 validation windows with strict per-window cutoffs; ExperimentIR 0.1.0 gates (time availability, split isolation, metrics, budget, baseline-same-protocol, execution authorization); 12 golden tests.
- **Verdict:** PASS (GPT 2026-10-26; HOLD lifted after Jaccard date-pair disclosure — all four window pairs lie at or before the seal boundary).

## 3. R1 — explicit features vs behavioural embeddings

| Domain | B0 rule | B1 GBDT | B2 GBDT+SVD-emb | B2−B1 |
|---|---|---|---|---|
| swire_ro W4 (n=4,605) | 0.3590 | 0.4255 | 0.4300 | +0.0045 (see paired CI in artifact) |
| H&M 10k active W4 (n=2,569) | 0.0393 | 0.0397 | 0.0389 | −0.0008 (not significant) |

- **Allowed conclusion (per GPT):** strong GBDT clearly beats rule baselines on swire_ro; behavioural embeddings add a small positive increment there; embeddings complement, not replace, explicit features. H&M S0 share of *scored* customers is 12.4% (not the 33.6% all-catalog figure).
- **Condition:** per-window out-of-time invariants must be archived before FINAL VERDICT; R1 vs R2 baseline numeric differences (B1 0.4246→0.4255, B2 0.4319→0.4300) must be explained.
  - **Explanation (2026-10-26):** protocol parameters are identical between R1 `run_r1_benchmark` and R2 `run_r2_orthogonal_benchmark` (250 trees, lr, seed 42, same candidate/feature construction code; zero diff in `r1_candidates.py`/`r1_features.py`/`metrics.py` between commits `a64b67c..d5a644f`). The ±0.001 cross-run differences are consistent with LightGBM multi-threaded non-determinism (`n_jobs=-1`, `deterministic` not forced). Adoption decisions therefore rely on same-run paired contrasts (e.g., swire Hybrid_B2 vs same-run B2: Δ=−0.0002, CI [−0.0015, +0.0010], not significant; H&M Hybrid_B2−B2 paired: Δ=−0.0003, p=0.804), which show **Hybrid does not beat same-protocol B2 in either domain**; the swire significant gain is Hybrid over B1 only.

## 4. R2 — 2×4+1 orthogonal matrix (encoder × loss, shared 256-negative protocol)

Track A MAP@12 (same-candidate rerank), frozen 8-epoch budget:

| Encoder | swire SS256 | H&M SS256 |
|---|---|---|
| MeanPool | **0.3584** | **0.0354** |
| GRU | 0.3390 | 0.0348 |
| SASRec | 0.2863 | 0.0337 |
| LiGR-SASRec (eSASRec-style) | 0.3281 | 0.0113* |
| HSTU-Math | 0.3263 | 0.0221* |

\* H&M BCE-256 arm shown under-trained by the epochs 8→24 convergence probe (HSTU_BCE256 +0.0175, LIGR_BCE256 +0.0050; SS256 arm ±0.0023). Per GPT: the 8-epoch BCE arm is **REJECT**ed as a final architecture ranking on H&M.

- **Order sensitivity (true − shuffled inference):** H&M GRU +0.0117 (S2 +0.0276), SASRec +0.0106 (S2 +0.0259), HSTU +0.0057, MeanPool 0.000 (positive control); swire GRU +0.0022, SASRec −0.0002, HSTU −0.0014.
- **Hybrid (Track C, zero-leakage OOF):** swire Hybrid_B1 vs B1 +0.0017, 95% CI [+0.0003, +0.0029], p=0.014 (multiplicity disclosure required); H&M not significant (p=0.224 / 0.52).
- **Allowed conclusions:** repeat-domain GBDT dominance; descriptive cross-domain order-sensitivity difference; H&M order sensitivity concentrated in S2; hybrid observed positive on swire, not observed on H&M.
- **Not claimed:** Attention provides independent predictive gain on H&M (NOT ESTABLISHED — order-sensitive models do not beat MeanPool); strong causal double-dissociation (INCONCLUSIVE); "SS256 always beats BCE256" (false).
- **Confirmation results (2026-10-26, `artifacts/eval/hm_enlarged_paired_ci.json`, `artifacts/eval/order_confirmation_arms.json`):**
  - **Exp 2 — enlarged frozen H&M sample (40k stratified @ cutoff 2020-09-08, seed 42; n=2,667 paired buyers, S2 n=801):** the exploratory 10k-sample S2 advantage of GRU over B1 **did not replicate**: paired GRU−B1 = −0.0061 [−0.0102, −0.0020] (ALL, p<0.001); S2 −0.0079 [−0.0159, +0.0004] (p=0.058). SAS−B1 S2 −0.0089 (p=0.048) and HSTU−B1 S2 −0.0109 (p=0.026) are significantly negative. GRU−MeanPool = −0.0004 [−0.0036, +0.0028] (p=0.822): sequence encoders do **not** beat the order-free pool under paired testing. Hybrid_B2−B2 = −0.0003 (p=0.804); Hybrid_B2−B1 = −0.0008 (p=0.628); B2−B1 = −0.0004 (p=0.764): neither embeddings nor neural OOF features add a measurable gain on H&M under the paired protocol.
  - **Exp 3 — four order arms (GRU & SASRec, SS256, 3 shuffle seeds, both domains):** H&M GRU A(true/true)=0.0343 > C(shuffled-train/shuffled-infer, matched) mean 0.0284 > B(true-train/shuffled-infer) mean 0.0270 >> D(reversed)=0.0175; SASRec A=0.0339 > C 0.0279 > B 0.0262 >> D 0.0190. Since matched retraining (C) recovers only part of the drop (A−C ≈ +0.006 vs A−B ≈ +0.0075), **true purchase order carries real, directionally-recent information on H&M** (A−D ≈ +0.015..0.017; S2 ≈ +0.034..0.037), while roughly 20% of the naive shuffle-drop is train–infer distribution mismatch. swire: A−B ≈ +0.0016 (GRU) / −0.0002 (SASRec) and C ≥ A (matched shuffling costs nothing: GRU C mean 0.3424 > A 0.3390; SASRec C 0.2943 > A 0.2863) → **full purchase order adds no information in the repeat domain**; the large reversed-arm drops (GRU D 0.2468) localize what these models actually use to the last-position recency structure rather than order content. swire SASRec's reversed score **exceeds** true order (D 0.3219 > A 0.2863) — NOT "direction-insensitive" (see GPT-verdict correction below).
  - **Net effect on C1:** the domain-conditioned *sensitivity* difference is confirmed and now mechanism-separated (mismatch vs true order vs recency direction). The stronger claim remains unchanged: on H&M, order utilization has **not** converted into an advantage over order-free MeanPool or strong GBDT (paired CIs above).
  - **GPT verdict on confirmations:** Exp2 PASS (exploratory S2 advantage withdrawn; S2 GRU−B1 CI crosses zero → "no confirmed advantage, negative point estimate", not "significantly worse"; SAS/HSTU S2 p=0.048/0.026 are unadjusted nominal significances). Exp3 PASS with required renames and scope: the D arm is a **"reversed-representation & direction-sensitivity diagnostic"**, not a pure recency orthogonal estimate (reversal also changes last-position semantics and per-position targets); swire SASRec's reversed score **exceeding** true order (+0.0356) forbids the claim "order- and direction-insensitive" — the anomaly is retained as an open observation pending inspection of last-position pooling, prediction head, and label alignment. Signed C1 wording (verbatim, GPT-approved):
    > 在本研究的两个零售数据域及冻结实验条件下，历史交易时间排列对不同模型的预测贡献表现出明显差异。在 H&M 时尚购买数据中，GRU 和 SASRec 使用真实时间排列的表现优于随机乱序训练与推理，且这种差异在历史较丰富的客户群中更明显。在 swire_ro 周期性补货数据中，完整原始购买顺序未显示出相对于匹配乱序训练的稳定优势；部分模型对序列方向或末端位置仍表现出明显敏感性。上述差异并未转化为序列模型相对于强 GBDT 或无序 MeanPool 的稳定总体预测优势，因此研究支持"时间排列的可利用性具有任务和模型依赖性"，但不支持"Attention 在探索型零售中具有独立预测优势"。
  - **Scope limits attached to all H&M confirmation numbers:** customers come from a pre-frozen 40,000-customer stratified sample; scored buyers n=2,667; article universe truncated to pre-cutoff Top-5,000. "No stable neural gain on H&M" is therefore **task-truncated**, not a statement about the full H&M catalog. Inverse-probability-weighted overall estimates (restoring full-catalog buyer mix, S2=75.0% of buyers vs 30.0% in the sample) are PENDING (§8).

## 5. R3 — single-GPU compute frontier

- AMP bf16/fp16: train speedup 1.14–1.32x, VRAM ratio ≈1.0, top-12 agreement 0.981–0.999 (output-changing, not exact-equivalent).
- SDPA bf16 (no torch.compile — BackendCompilerFailed on Windows, disclosed per-row): SASRec inference 61.8k→128.3k users/s (2.08x); HSTU eager throughput collapses super-linearly with L (19.5k→0.99k seq/s from L=32→256).
- Capacity sweep d64L2→d256L4: ≤2.9GB VRAM everywhere; the 20GB card is not memory-bound at this scale.
- **Allowed:** AMP/SDPA are worthwhile optimization candidates on this workload; measured speedups are implementation- and precision-dependent.
- **Not claimed:** compute is THE bottleneck (INCONCLUSIVE); cross-hardware/framework superiority (untested).

## 6. Threats to validity and disclosures

- H&M R2 primary was a 10k active-customer exploratory sample; enlarged 40k paired confirmation completed 2026-10-26 (exploratory S2 advantage withdrawn; task-truncation scope limits in §4). HSTU-Math is a from-scratch mathematical reimplementation (reference-consistency evidence PENDING, §8).
- Multiple comparisons across the 2×4+1 matrix and hybrid variants; p-values uncorrected.
- Windows-only torch.compile failure; single seed per neural run in the primary matrix (shuffle arms use 3 seeds).
- W5 remains sealed; all model/confirmation choices were fixed before any W5 peek (none occurred).

## 7. Prohibited assertions (standing, per GPT verdict)

"Attention math beats traditional math in exploratory retail" / "HSTU unsuitable for recsys" / "repeat share causally determines Hybrid gain" / "2022 failure was GPU architecture" / "more model or compute cannot help" — none of these may appear as proven claims in any output derived from this report.

## 8. PENDING verification items (research limitations, per GPT verdict — not blocking finalization)

1. **REPORT PROVENANCE PENDING** — per-window out-of-time invariant acceptance record (train-label end date, embedding fit cutoff, candidate generation cutoff, feature max-available time per evaluated window). W4 frozen comparisons are signed; W2–W4 must not be advertised as independently verified walk-forward generalization until archived.
2. **ARCHITECTURE IDENTITY PENDING** — HSTU-Math is a from-scratch mathematical reimplementation. All HSTU rows are "this project's HSTU-Math implementation as measured", NOT NVIDIA official HSTU performance, and cannot be used to judge the HSTU architecture itself. Upgrading requires mask/position-bias/aggregation/forward-gradient reference-consistency evidence.
3. **COMPUTE FIDELITY PENDING** — R3 is signed as an engineering performance benchmark for this workload only; no exact-equivalence claim (top-12 agreement 0.981–0.999 is output-changing). Per-mode MAP@12 difference CIs not yet produced.
4. **IPW OVERALL ESTIMATE PENDING** — inverse-probability-weighted MAP@12 and paired differences on the 40k sample to restore the full-catalog buyer mix (requires persisting per-customer AP in a rerun; current per-customer APs were not persisted).
5. **LGBM DETERMINISM DIAGNOSTIC PENDING** — the R1/R2 baseline delta attribution to multi-threaded non-determinism is a reasonable hypothesis, not a proven fact; minimal diagnostic = same data/feature/candidate hashes and version, repeated n_jobs=-1 runs vs deterministic CPU runs (`deterministic=true` + `force_col_wise`/`force_row_wise`). Deterministic-config results must not replace frozen experiment numbers.

## 9. Hybrid engineering verdict and central signed conclusion

- **Hybrid engineering adoption: NOT RECOMMENDED UNDER CURRENT EVIDENCE** — in neither domain does the neural-sequential OOF fusion beat same-protocol B2 (swire Δ=−0.0002 ns; H&M Δ=−0.0003 ns). Current evidence does not justify maintaining a separate neural sequence model in two serving pipelines for this increment. Not claimable as "neural information is fully redundant" — only that the increment was not validated under current sample, candidate pool, budget, and fusion implementation.

**Central conclusion (GPT-signed wording):**

> 在两个业务结构截然不同的零售数据域中，新一代序列模型能够对历史时间排列产生不同程度的响应，但这种能力并不自动转化为相对于成熟统计特征、行为 Embedding 和 GBDT 的额外预测价值。H&M 的确认实验未能复现早期小样本中观察到的序列模型优势；swire_ro 中，行为 Embedding 提供了有限但可测量的增量，进一步融合神经评分没有稳定收益。计算实现优化可以降低部分模型的执行成本，却不能替代对学习目标、数据表示、时间语义与统计泛化的验证。

Per GPT closing directive: stop open-ended model search; carry the three evidence gaps as explicit verification items; any future work starts from a new falsifiable hypothesis, not from additional compute allocated to make HSTU or Attention win.
