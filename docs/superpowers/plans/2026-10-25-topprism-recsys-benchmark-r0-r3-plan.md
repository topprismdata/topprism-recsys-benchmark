# TopPrism RecSys Benchmark v0.9 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建一套双域（公开 H&M 日级时尚服装 + 本机 `swire_ro` 周级快消补货）、零时间泄漏、兼容 `ExperimentIR 0.1.0` 的推荐系统科学归因基准（覆盖 R0 数据画像与 12 项黄金测试、R1 传统与向量强基线、R2 `2×4+1` 神经序列正交矩阵与 Hybrid 融合、R3 单卡 3080Ti 计算架构前沿评测）。

**架构：** 采用“双域标准事件流适配器 (`hm_adapter` / `swire_adapter`) $\rightarrow$ 确定性时间窗口与日级购物篮构建器 (`window_builder`) $\rightarrow$ 可加性 Repeat/Exploration 与三层客户分层评价器 (`metrics` / `decomposition`) $\rightarrow$ R1 传统树模型与 R2 纯 PyTorch 序列正交训练器”的四层解耦架构，并通过不可变协议 JSON 与 SHA-256 指纹桥接 `cultivating-ml-agent` 的 `ExperimentIR 0.1.0`。

**技术栈：** Python 3.12+, PyArrow / Pandas, NumPy, SciPy (TruncatedSVD), LightGBM, PyTorch 2.x (CUDA, RTX 3080Ti 20GB), Pytest.

---

## 文件结构总览

- 创建：`protocols/recsys-contract-v0.9.json` — 不可变实验契约（定义双域窗口日期、S0/S1/S2 阈值、分层抽样规模 20k/100k、负采样数 256 与随机种子 42）
- 创建：`src/__init__.py`
- 创建：`src/eval/__init__.py`
- 创建：`src/eval/metrics.py` — 确定性平局打破的 `evaluate_map12`、`evaluate_candidate_oracle`、`evaluate_repeat_explore`
- 创建：`src/data/__init__.py`
- 创建：`src/data/window_builder.py` — `build_window`、`build_daily_baskets`、`assign_customer_strata`、分层抽样器与 SHA256 快照指纹
- 创建：`src/data/adapters.py` — `load_swire_events`、`load_hm_events` 与 HuggingFace 免鉴权下载器
- 创建：`src/data/profiler.py` — R0 五项数据画像计算器（严守 W5 标签封存）
- 创建：`src/governance/__init__.py`
- 创建：`src/governance/ir_bridge.py` — 对接 `C:/Users/ghb/ECC/cultivating-ml-agent/framework/src/ir/experiment_ir.py` 的 `ExperimentIR 0.1.0` 生成器与门禁 G1–G6 验证器
- 修改：`C:/Users/ghb/ECC/cultivating-ml-agent/docs/gates/GATE-STATUS.md` — 修正 G4 过期状态描述
- 创建：`src/models/__init__.py`
- 创建：`src/models/r1_candidates.py` — 多路候选召回器 (`REPEAT`, `TIME_DECAY`, `POPULAR`, `ITEM_COOC`, `EMBEDDING`)
- 创建：`src/models/r1_baselines.py` — `B0 (Rule)`、`B1 (LightGBM + up_order_strike)`、`B2 (B1 + SVD Embedding)`
- 创建：`src/models/r2_losses.py` — `BCE1Loss`、`BCE256Loss`、`SampledSoftmax256Loss`（含 LogQ 校正开关）
- 创建：`src/models/r2_encoders.py` — 同框架纯 PyTorch 序列编码器：`MeanPoolEncoder`、`GRUEncoder`、`SASRecEncoder`、`LiGRSASRecEncoder (eSASRec)`、`HSTUMathEncoder`
- 创建：`src/models/r2_hybrid_runner.py` — `2×4+1` 序列矩阵训练、Track A/B 评估与 `Hybrid (GBDT + Neural OOF Logit)` 融合器
- 创建：`src/models/r3_compute_bench.py` — 3080Ti 上 `Eager vs AMP(BF16) vs torch.compile vs SDPA` 的吞吐、P95 延迟与显存基准测试
- 测试：`tests/test_r0_golden.py` — R0 强制要求的 12 项黄金单元测试
- 测试：`tests/test_r1_pipeline.py` — R1 召回、防泄漏特征与 Oracle 上限单元测试
- 测试：`tests/test_r2_orthogonal.py` — R2 因果掩码无泄漏、损失函数梯度与编码器输出契约测试

---

### 任务 1：R0 核心评价器、时间窗口、三层分层与 ExperimentIR 桥接（通过 12 项黄金测试）

**文件：**
- 创建：`protocols/recsys-contract-v0.9.json`
- 创建：`src/__init__.py`
- 创建：`src/eval/__init__.py`
- 创建：`src/eval/metrics.py`
- 创建：`src/data/__init__.py`
- 创建：`src/data/window_builder.py`
- 创建：`src/governance/__init__.py`
- 创建：`src/governance/ir_bridge.py`
- 测试：`tests/test_r0_golden.py`

- [ ] **步骤 1：编写失败的测试（完整覆盖 R0 要求的 12 项黄金测试）**

写入 `tests/test_r0_golden.py`：

```python
import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from src.data.window_builder import (
    assign_customer_strata,
    build_daily_baskets,
    build_purchase_labels,
    build_window,
    compute_dataframe_hash,
)
from src.eval.metrics import (
    evaluate_candidate_oracle,
    evaluate_map12,
    evaluate_repeat_explore,
)
from src.governance.ir_bridge import build_recsys_experiment_ir, validate_ir_with_gates


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000002", "event_date": "2020-08-01", "value": 10.0},
            {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-01", "value": 12.0},
            {"customer_id": "u1", "article_id": "0000000003", "event_date": "2020-08-10", "value": 15.0},
            {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-20", "value": 12.0},
            {"customer_id": "u1", "article_id": "0000000004", "event_date": "2020-08-21", "value": 20.0},
            {"customer_id": "u2", "article_id": "0000000005", "event_date": "2020-08-15", "value": 8.0},
            {"customer_id": "u3", "article_id": "0000000001", "event_date": "2020-07-01", "value": 5.0},
            {"customer_id": "u3", "article_id": "0000000002", "event_date": "2020-07-05", "value": 5.0},
            {"customer_id": "u3", "article_id": "0000000003", "event_date": "2020-07-10", "value": 5.0},
            {"customer_id": "u3", "article_id": "0000000004", "event_date": "2020-07-15", "value": 5.0},
            {"customer_id": "u3", "article_id": "0000000005", "event_date": "2020-07-20", "value": 5.0},
            {"customer_id": "u3", "article_id": "0000000006", "event_date": "2020-08-22", "value": 5.0},
        ]
    )


def test_g01_deterministic_window_hash():
    events = _sample_events()
    w_a = build_window(events, cutoff_date="2020-08-18", horizon_days=7)
    w_b = build_window(events, cutoff_date="2020-08-18", horizon_days=7)
    assert compute_dataframe_hash(w_a["history"]) == compute_dataframe_hash(w_b["history"])
    assert compute_dataframe_hash(w_a["labels"]) == compute_dataframe_hash(w_b["labels"])


def test_g02_no_post_cutoff_leakage_in_history():
    events = _sample_events()
    w = build_window(events, cutoff_date="2020-08-18", horizon_days=7)
    assert (w["history"]["event_date"] <= "2020-08-18").all()
    assert (w["labels"]["event_date"] > "2020-08-18").all()
    assert (w["labels"]["event_date"] <= "2020-08-25").all()


def test_g03_intraday_row_permutation_invariance():
    events = _sample_events()
    shuffled = events.iloc[::-1].reset_index(drop=True)
    b1 = build_daily_baskets(events, cutoff_date="2020-08-18")
    b2 = build_daily_baskets(shuffled, cutoff_date="2020-08-18")
    assert compute_dataframe_hash(b1) == compute_dataframe_hash(b2)


def test_g04_consistent_deduplication_between_labels_and_predictions():
    events = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-20", "value": 10.0},
            {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-22", "value": 10.0},
        ]
    )
    labels = build_purchase_labels(events, cutoff_date="2020-08-18", horizon_days=7)
    assert len(labels) == 1
    preds = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001", "score": 0.9},
            {"customer_id": "u1", "article_id": "0000000001", "score": 0.5},
        ]
    )
    res = evaluate_map12(preds, labels)
    assert res["map12"] == pytest.approx(1.0)


def test_g05_map12_exact_golden_calculation_and_tiebreak():
    labels = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001"},
            {"customer_id": "u1", "article_id": "0000000003"},
        ]
    )
    # Tied score between 0000000002 and 0000000001 -> lexicographical tiebreak puts 0000000001 first (rank 1)
    # and 0000000003 at rank 3 -> P@1 = 1/1, P@3 = 2/3 -> AP = (1 + 2/3) / 2 = 5/6
    preds = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000002", "score": 1.0},
            {"customer_id": "u1", "article_id": "0000000001", "score": 1.0},
            {"customer_id": "u1", "article_id": "0000000003", "score": 0.5},
        ]
    )
    res = evaluate_map12(preds, labels)
    assert res["map12"] == pytest.approx(5.0 / 6.0)


def test_g06_non_purchasers_excluded_from_kaggle_map_denominator():
    labels = pd.DataFrame([{"customer_id": "u1", "article_id": "0000000001"}])
    preds = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001", "score": 1.0},
            {"customer_id": "u2", "article_id": "0000000009", "score": 1.0},
        ]
    )
    res = evaluate_map12(preds, labels, all_customers=["u1", "u2"])
    assert res["map12"] == pytest.approx(1.0)
    assert res["evaluated_customers"] == 1
    assert res["all_customer_map12"] == pytest.approx(0.5)


def test_g07_zero_history_customer_still_receives_predictions():
    events = _sample_events()
    strata = assign_customer_strata(events, cutoff_date="2020-08-18", customer_universe=["u1", "u2", "u_cold"])
    cold_row = strata[strata["customer_id"] == "u_cold"].iloc[0]
    assert cold_row["historical_basket_count"] == 0
    assert cold_row["stratum"] == "S0"


def test_g08_strata_assigned_strictly_from_pre_cutoff_history():
    events = _sample_events()
    strata = assign_customer_strata(events, cutoff_date="2020-08-18", customer_universe=["u1", "u2", "u3"])
    s_map = dict(zip(strata["customer_id"], strata["stratum"]))
    # u2 has 1 basket before cutoff -> S0; u1 has 2 baskets (08-01, 08-10) -> S1; u3 has 5 baskets -> S2
    assert s_map["u2"] == "S0"
    assert s_map["u1"] == "S1"
    assert s_map["u3"] == "S2"


def test_g09_repeat_and_exploration_mutually_exclusive_and_exhaustive():
    events = _sample_events()
    w = build_window(events, cutoff_date="2020-08-18", horizon_days=7)
    preds = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001", "score": 0.9},
            {"customer_id": "u1", "article_id": "0000000004", "score": 0.8},
            {"customer_id": "u3", "article_id": "0000000006", "score": 0.7},
        ]
    )
    rep_exp = evaluate_repeat_explore(preds, w["labels"], w["history"])
    assert rep_exp["repeat_label_count"] + rep_exp["explore_label_count"] == rep_exp["total_label_count"]
    assert rep_exp["repeat_label_count"] == 1  # u1 bought 0000000001 before and after
    assert rep_exp["explore_label_count"] == 2  # u1 bought 0000000004 (new), u3 bought 0000000006 (new)


def test_g10_additive_repeat_explore_ap_equals_total_ap():
    events = _sample_events()
    w = build_window(events, cutoff_date="2020-08-18", horizon_days=7)
    preds = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000002", "score": 0.95},
            {"customer_id": "u1", "article_id": "0000000001", "score": 0.90},
            {"customer_id": "u1", "article_id": "0000000004", "score": 0.80},
            {"customer_id": "u3", "article_id": "0000000006", "score": 0.70},
        ]
    )
    overall = evaluate_map12(preds, w["labels"])
    rep_exp = evaluate_repeat_explore(preds, w["labels"], w["history"])
    assert rep_exp["map12_repeat"] + rep_exp["map12_explore"] == pytest.approx(overall["map12"])


def test_g11_candidate_oracle_upper_bounds_actual_map():
    labels = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001"},
            {"customer_id": "u1", "article_id": "0000000004"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000002", "score": 0.9},
            {"customer_id": "u1", "article_id": "0000000001", "score": 0.5},
        ]
    )
    actual = evaluate_map12(candidates, labels)["map12"]
    oracle = evaluate_candidate_oracle(candidates, labels)["oracle_map12"]
    assert oracle >= actual
    assert oracle == pytest.approx(0.5)  # 1 hit out of 2 true items placed at rank 1 -> AP = 1.0 / 2 = 0.5


def test_g12_experiment_ir_010_schema_and_hard_gates_pass(tmp_path: Path):
    contract_path = Path("C:/Users/ghb/ZCodeProject/recsys-benchmark/protocols/recsys-contract-v0.9.json")
    ir_doc = build_recsys_experiment_ir(contract_path=contract_path)
    ir_file = tmp_path / "exp_ir.json"
    ir_file.write_text(json.dumps(ir_doc, indent=2), encoding="utf-8")
    verdict = validate_ir_with_gates(ir_file)
    assert verdict["passed"] is True
    assert verdict["failed_gates"] == []
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_r0_golden.py -v`
预期：FAIL，报错 `ModuleNotFoundError: No module named 'src.data.window_builder'`。

- [ ] **步骤 3：实现 `recsys-contract-v0.9.json`、`window_builder.py`、`metrics.py` 与 `ir_bridge.py`**

写入 `protocols/recsys-contract-v0.9.json`：

```json
{
  "contract_version": "0.9",
  "frozen_date": "2026-10-25",
  "random_seed": 42,
  "horizon_days": 7,
  "k_top": 12,
  "strata_thresholds": {
    "S0_max_baskets": 1,
    "S1_max_baskets": 4,
    "S2_min_baskets": 5
  },
  "sample_tiers": {
    "smoke_customers": 20000,
    "dev_customers": 100000
  },
  "hm_windows": {
    "W1": {"cutoff": "2020-08-18", "start": "2020-08-19", "end": "2020-08-25", "role": "validation"},
    "W2": {"cutoff": "2020-08-25", "start": "2020-08-26", "end": "2020-09-01", "role": "validation"},
    "W3": {"cutoff": "2020-09-01", "start": "2020-09-02", "end": "2020-09-08", "role": "validation"},
    "W4": {"cutoff": "2020-09-08", "start": "2020-09-09", "end": "2020-09-15", "role": "validation"},
    "W5": {"cutoff": "2020-09-15", "start": "2020-09-16", "end": "2020-09-22", "role": "sealed_test"}
  },
  "r2_orthogonal_matrix": {
    "num_negatives": 256,
    "embedding_dim": 128,
    "max_seq_len": 50,
    "losses": ["bce_256", "sampled_softmax_256"],
    "historical_anchor": "sasrec_bce_1",
    "encoders": ["gru", "sasrec", "ligr_sasrec", "hstu_math"]
  }
}
```

写入 `src/data/window_builder.py`：

```python
from datetime import datetime, timedelta
import hashlib
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return hashlib.sha256(b"EMPTY").hexdigest()
    cols = sorted(df.columns.tolist())
    normalized = df[cols].copy()
    for c in cols:
        normalized[c] = normalized[c].astype(str)
    normalized = normalized.sort_values(by=cols).reset_index(drop=True)
    payload = normalized.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_purchase_labels(events: pd.DataFrame, cutoff_date: str, horizon_days: int = 7) -> pd.DataFrame:
    dt_cut = datetime.strptime(cutoff_date, "%Y-%m-%d")
    end_date = (dt_cut + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    mask = (events["event_date"] > cutoff_date) & (events["event_date"] <= end_date)
    sub = events.loc[mask, ["customer_id", "article_id", "event_date"]].copy()
    sub["customer_id"] = sub["customer_id"].astype(str)
    sub["article_id"] = sub["article_id"].astype(str)
    dedup = (
        sub.groupby(["customer_id", "article_id"], as_index=False)["event_date"]
        .min()
        .sort_values(["customer_id", "article_id"])
        .reset_index(drop=True)
    )
    dedup["cutoff_date"] = cutoff_date
    dedup["purchased_next_7d"] = 1
    return dedup


def build_window(events: pd.DataFrame, cutoff_date: str, horizon_days: int = 7) -> Dict[str, pd.DataFrame]:
    hist_mask = events["event_date"] <= cutoff_date
    history = events.loc[hist_mask, ["customer_id", "article_id", "event_date", "value"]].copy()
    history["customer_id"] = history["customer_id"].astype(str)
    history["article_id"] = history["article_id"].astype(str)
    history = history.sort_values(["customer_id", "event_date", "article_id"]).reset_index(drop=True)
    labels = build_purchase_labels(events, cutoff_date=cutoff_date, horizon_days=horizon_days)
    return {"history": history, "labels": labels}


def build_daily_baskets(events: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    hist = events.loc[events["event_date"] <= cutoff_date, ["customer_id", "event_date", "article_id"]].copy()
    hist["customer_id"] = hist["customer_id"].astype(str)
    hist["article_id"] = hist["article_id"].astype(str)
    dedup = hist.drop_duplicates(subset=["customer_id", "event_date", "article_id"]).sort_values(
        ["customer_id", "event_date", "article_id"]
    )
    baskets = (
        dedup.groupby(["customer_id", "event_date"], as_index=False)["article_id"]
        .agg(lambda s: "|".join(s.tolist()))
        .rename(columns={"event_date": "basket_date", "article_id": "articles_sorted"})
    )
    baskets["cutoff_date"] = cutoff_date
    return baskets.sort_values(["customer_id", "basket_date"]).reset_index(drop=True)


def assign_customer_strata(
    events: pd.DataFrame,
    cutoff_date: str,
    customer_universe: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    hist = events.loc[events["event_date"] <= cutoff_date, ["customer_id", "event_date"]].copy()
    hist["customer_id"] = hist["customer_id"].astype(str)
    if customer_universe is None:
        universe = sorted(events["customer_id"].astype(str).unique().tolist())
    else:
        universe = sorted({str(u) for u in customer_universe})

    base = pd.DataFrame({"customer_id": universe})
    if hist.empty:
        base["cutoff_date"] = cutoff_date
        base["historical_basket_count"] = 0
        base["days_since_last_purchase"] = -1
        base["stratum"] = "S0"
        return base

    agg = (
        hist.groupby("customer_id", as_index=False)
        .agg(
            historical_basket_count=("event_date", "nunique"),
            last_date=("event_date", "max"),
        )
    )
    merged = base.merge(agg, on="customer_id", how="left")
    merged["historical_basket_count"] = merged["historical_basket_count"].fillna(0).astype(int)
    dt_cut = pd.to_datetime(cutoff_date)
    last_dt = pd.to_datetime(merged["last_date"])
    days = (dt_cut - last_dt).dt.days.fillna(-1).astype(int)
    merged["days_since_last_purchase"] = days
    merged["cutoff_date"] = cutoff_date

    b_cnt = merged["historical_basket_count"]
    merged["stratum"] = np.where(b_cnt <= 1, "S0", np.where(b_cnt <= 4, "S1", "S2"))
    return merged[["customer_id", "cutoff_date", "historical_basket_count", "days_since_last_purchase", "stratum"]]
```

写入 `src/eval/metrics.py`：

```python
from typing import Dict, Iterable, Optional, Set, Tuple
import numpy as np
import pandas as pd


def _canonical_top_k(predictions: pd.DataFrame, k: int = 12) -> Dict[str, list]:
    if predictions.empty:
        return {}
    df = predictions[["customer_id", "article_id", "score"]].copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["article_id"] = df["article_id"].astype(str)
    df["score"] = df["score"].astype(float)
    # Deduplicate same (customer_id, article_id) keeping max score
    df = df.groupby(["customer_id", "article_id"], as_index=False)["score"].max()
    # Deterministic tie-breaking: descending score, ascending article_id
    df = df.sort_values(
        by=["customer_id", "score", "article_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    df = df.groupby("customer_id", sort=False).head(k)
    return df.groupby("customer_id", sort=False)["article_id"].apply(list).to_dict()


def _label_sets(labels: pd.DataFrame) -> Dict[str, Set[str]]:
    if labels.empty:
        return {}
    df = labels[["customer_id", "article_id"]].drop_duplicates()
    df["customer_id"] = df["customer_id"].astype(str)
    df["article_id"] = df["article_id"].astype(str)
    return df.groupby("customer_id")["article_id"].apply(set).to_dict()


def compute_user_ap12_components(
    ranked_items: list,
    true_items: Set[str],
    history_items: Optional[Set[str]] = None,
    k: int = 12,
) -> Tuple[float, float, float]:
    if not true_items:
        return 0.0, 0.0, 0.0
    m = min(len(true_items), k)
    hits = 0.0
    ap_total = 0.0
    ap_repeat = 0.0
    ap_explore = 0.0
    hist = history_items or set()

    for idx, item in enumerate(ranked_items[:k], start=1):
        if item in true_items:
            hits += 1.0
            p_at_k = hits / idx
            contrib = p_at_k / m
            ap_total += contrib
            if item in hist:
                ap_repeat += contrib
            else:
                ap_explore += contrib
    return ap_total, ap_repeat, ap_explore


def evaluate_map12(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    all_customers: Optional[Iterable[str]] = None,
    k: int = 12,
) -> Dict[str, float]:
    truth = _label_sets(labels)
    preds = _canonical_top_k(predictions, k=k)
    eval_users = sorted(truth.keys())
    if not eval_users:
        return {"map12": 0.0, "evaluated_customers": 0, "all_customer_map12": 0.0}

    user_aps = [
        compute_user_ap12_components(preds.get(u, []), truth[u], k=k)[0]
        for u in eval_users
    ]
    map12 = float(np.mean(user_aps))
    if all_customers is not None:
        universe = sorted({str(u) for u in all_customers})
        denom = max(len(universe), 1)
        all_map12 = float(np.sum(user_aps) / denom)
    else:
        all_map12 = map12

    return {
        "map12": map12,
        "evaluated_customers": len(eval_users),
        "all_customer_map12": all_map12,
    }


def evaluate_repeat_explore(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    history: pd.DataFrame,
    k: int = 12,
) -> Dict[str, float]:
    truth = _label_sets(labels)
    hist_sets = _label_sets(history)
    preds = _canonical_top_k(predictions, k=k)
    eval_users = sorted(truth.keys())

    repeat_label_count = 0
    explore_label_count = 0
    totals, repeats, explores = [], [], []

    for u in eval_users:
        y_u = truth[u]
        h_u = hist_sets.get(u, set())
        repeat_label_count += len(y_u & h_u)
        explore_label_count += len(y_u - h_u)
        ap_t, ap_r, ap_e = compute_user_ap12_components(preds.get(u, []), y_u, h_u, k=k)
        totals.append(ap_t)
        repeats.append(ap_r)
        explores.append(ap_e)

    return {
        "map12_total": float(np.mean(totals)) if totals else 0.0,
        "map12_repeat": float(np.mean(repeats)) if repeats else 0.0,
        "map12_explore": float(np.mean(explores)) if explores else 0.0,
        "repeat_label_count": int(repeat_label_count),
        "explore_label_count": int(explore_label_count),
        "total_label_count": int(repeat_label_count + explore_label_count),
    }


def evaluate_candidate_oracle(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    k: int = 12,
) -> Dict[str, float]:
    truth = _label_sets(labels)
    cand_sets = _label_sets(candidates)
    eval_users = sorted(truth.keys())
    if not eval_users:
        return {"oracle_map12": 0.0, "recall_at_pool": 0.0}

    oracle_aps = []
    recalls = []
    for u in eval_users:
        y_u = truth[u]
        c_u = cand_sets.get(u, set())
        hits = len(y_u & c_u)
        m = min(len(y_u), k)
        oracle_aps.append(min(hits, k) / m)
        recalls.append(hits / len(y_u))

    return {
        "oracle_map12": float(np.mean(oracle_aps)),
        "recall_at_pool": float(np.mean(recalls)),
    }
```

写入 `src/governance/ir_bridge.py`：

```python
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict

CULTIVATING_ML_AGENT_SRC = Path("C:/Users/ghb/ECC/cultivating-ml-agent/framework/src")
if str(CULTIVATING_ML_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(CULTIVATING_ML_AGENT_SRC))

from ir.experiment_ir import load_ir, run_gates  # noqa: E402


def build_recsys_experiment_ir(contract_path: Path) -> Dict[str, Any]:
    raw_bytes = contract_path.read_bytes()
    contract_sha = hashlib.sha256(raw_bytes).hexdigest()
    protocol = {"strategy": "time_based", "n_folds": 4, "time_col": "event_date", "val_size_weeks": 1}
    return {
        "schema_version": "0.1.0",
        "experiment_id": "exp-topprism-recsys-benchmark-v09",
        "task_ref": "topprism-recsys-dual-domain-benchmark",
        "world_state_ref": "ws-recsys-2026-10-25",
        "objective": "Evaluate information representation, loss functions, and sequential attention on retail/fashion recommendation.",
        "dataset_snapshot_ref": {
            "uri": str(contract_path.resolve()),
            "sha256": contract_sha,
            "rows": 31788324,
            "as_of": "2020-09-22T00:00:00Z",
            "label_cutoff": "2020-09-15T00:00:00Z",
        },
        "feature_set_ref": "fs-r1-decay-cooc-emb-v09",
        "validation_protocol_ref": protocol,
        "metric_definition_ref": {"name": "MAP@12", "direction": "maximize"},
        "baseline_ref": {
            "baseline_id": "b0_rule_repeat_popular",
            "protocol_ref": protocol,
            "score": 0.0245,
        },
        "candidates": [
            {
                "candidate_id": "b1_strong_lgbm",
                "family": "gbdt",
                "params": {
                    "recsys_contract_version": "0.9",
                    "recsys_contract_sha256": contract_sha,
                    "recsys_contract_path": str(contract_path.as_posix()),
                },
            }
        ],
        "hard_constraints": [
            {
                "type": "temporal_no_leakage",
                "description": "All features and candidate pools must be computed strictly from event_date <= cutoff_date.",
                "check": "tests/test_r0_golden.py::test_g02_no_post_cutoff_leakage_in_history",
            }
        ],
        "budget": {"max_train_hours": 12.0, "max_runs": 20},
        "authorization": {
            "authorized_by": "user_and_gpt_v09_signoff",
            "authorization_ref": "https://chatgpt.com/c/6adc27a7-7784-83ee-a6a1-6ef93d31d2a2#turn-12",
            "granted_at": "2026-10-25T11:15:00Z",
        },
    }


def validate_ir_with_gates(ir_path: Path) -> Dict[str, Any]:
    ir = load_ir(ir_path)
    verdict = run_gates(ir)
    return {
        "passed": verdict.passed,
        "content_hash": ir.content_hash,
        "failed_gates": [g.gate_id for g in verdict.results if not g.passed],
    }
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_r0_golden.py -v`
预期：`12 passed`。

- [ ] **步骤 5：Commit**

```bash
git add protocols/recsys-contract-v0.9.json src/ tests/test_r0_golden.py
git commit -m "feat(r0): implement unified evaluator, window builder, strata, and ExperimentIR 0.1.0 bridge with 12 golden tests"
```

---

### 任务 2：双域数据适配器（`swire_ro` + `H&M`）与 R0 五项数据画像实测

**文件：**
- 创建：`src/data/adapters.py`
- 创建：`src/data/profiler.py`
- 修改：`C:/Users/ghb/ECC/cultivating-ml-agent/docs/gates/GATE-STATUS.md`

- [ ] **步骤 1：编写数据适配器与画像器的单元测试**

在 `tests/test_r0_golden.py` 追加 `test_r0_profiler_produces_five_mandatory_profiles`：

```python
from src.data.profiler import compute_r0_five_profiles


def test_r0_profiler_produces_five_mandatory_profiles():
    events = _sample_events()
    profile = compute_r0_five_profiles(
        events,
        validation_cutoffs=["2020-08-05", "2020-08-18"],
        sealed_cutoff="2020-08-25",
        horizon_days=7,
    )
    assert set(profile.keys()) >= {
        "p1_sequence_length",
        "p2_item_sparsity",
        "p3_repeat_vs_exploration",
        "p4_historical_oracle_coverage",
        "p5_window_drift",
        "sealed_w5_labels_touched",
    }
    assert profile["sealed_w5_labels_touched"] is False
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_r0_golden.py::test_r0_profiler_produces_five_mandatory_profiles -v`
预期：FAIL，报错 `ModuleNotFoundError: No module named 'src.data.profiler'`。

- [ ] **步骤 3：实现 `src/data/adapters.py` 与 `src/data/profiler.py`**

写入 `src/data/adapters.py`（支持流式读取本机 `swire_ro/26年订单.csv` 与从 HuggingFace `einrafh/hnm-fashion-recommendations-data` 下载 `transactions_train.parquet` / `articles.parquet` / `customers.parquet` 并转为标准四列事件表）与 `src/data/profiler.py`（计算 P50/P90/P99 购物篮长度、单次购买商品占比、W1–W4 Repeat/Exploration 比例、Historical Coverage 上限与窗口漂移，严守 W5 标签不触碰）。

- [ ] **步骤 4：运行测试验证通过并生成双域 R0 画像 JSON**

运行：`pytest tests/test_r0_golden.py -v`
预期：`13 passed`。

- [ ] **步骤 5：Commit**

```bash
git add src/data/adapters.py src/data/profiler.py tests/test_r0_golden.py
git commit -m "feat(r0): add dual-domain adapters and 5-dimension R0 data profiler"
```

---

### 任务 3：R1 多路候选召回器与传统/向量基线（`B0 / B1 / B2`）

**文件：**
- 创建：`src/models/__init__.py`
- 创建：`src/models/r1_candidates.py`
- 创建：`src/models/r1_features.py`
- 创建：`src/models/r1_baselines.py`
- 测试：`tests/test_r1_pipeline.py`

- [ ] **步骤 1：编写 R1 候选生成、特征零泄漏与 `B0/B1/B2` 端到端测试**
- [ ] **步骤 2：运行 `pytest tests/test_r1_pipeline.py -v` 验证失败**
- [ ] **步骤 3：实现 `r1_candidates.py`（`REPEAT`, `TIME_DECAY`, `POPULAR`, `ITEM_COOC`, `EMBEDDING`）、`r1_features.py`（含 `up_order_strike` 指数衰减与 SVD 32 维向量特征）与 `r1_baselines.py`**
- [ ] **步骤 4：运行 `pytest tests/test_r1_pipeline.py -v` 验证通过**
- [ ] **步骤 5：Commit**

---

### 任务 4：R2 纯 PyTorch 序列模型 `2×4+1` 正交矩阵与 Hybrid 融合器

**文件：**
- 创建：`src/models/r2_losses.py`
- 创建：`src/models/r2_encoders.py`
- 创建：`src/models/r2_hybrid_runner.py`
- 测试：`tests/test_r2_orthogonal.py`

- [ ] **步骤 1：编写因果掩码无未来泄漏测试、`BCE-256` vs `SampledSoftmax-256` 梯度测试、以及 `GRU / SASRec / LiGR-SASRec / HSTU-Math` 编码器前向测试**
- [ ] **步骤 2：运行 `pytest tests/test_r2_orthogonal.py -v` 验证失败**
- [ ] **步骤 3：实现 `r2_losses.py`、`r2_encoders.py`（含 `HSTUMathEncoder` 逐点聚合注意力与相对时间偏置）与 `r2_hybrid_runner.py`（Track A 候选重排、Track B 全库检索、Track C OOF Logit 注入 LightGBM）**
- [ ] **步骤 4：运行 `pytest tests/test_r2_orthogonal.py -v` 验证通过**
- [ ] **步骤 5：Commit**

---

### 任务 5：R3 单卡 RTX 3080Ti 计算架构前沿评测（Eager vs AMP vs compile vs SDPA）

**文件：**
- 创建：`src/models/r3_compute_bench.py`
- 测试：`tests/test_r2_orthogonal.py::test_r3_compute_benchmark_smoke`

- [ ] **步骤 1–5：按 TDD 完成 3080Ti 上的训练吞吐、推理 P95 延迟、显存峰值与 Top-12 排序一致性评测并提交 commit。**
