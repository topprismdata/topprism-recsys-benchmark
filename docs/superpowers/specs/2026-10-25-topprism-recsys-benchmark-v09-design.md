# TopPrism RecSys Benchmark v0.9 — 双域推荐系统科学归因与工程基准规格说明

> **日期**：2026-10-25
> **状态**：Draft — 经 Superpowers Brainstorming 生成并完成自检，待用户与 GPT 审阅
> **技能工作流**：`superpowers:brainstorming` $\rightarrow$ `superpowers:writing-plans` $\rightarrow$ `superpowers:subagent-driven-development`
> **工程根目录**：`C:/Users/ghb/ZCodeProject/recsys-benchmark/`

---

## 1. 研究目标与核心问题

### 1.1 首要科学问题
在严格统一的业务预测目标、零时间泄漏切分、相同候选池与可控计算资源边界（单卡 NVIDIA RTX 3080Ti 20GB VRAM）下：
1. **信息表示增量（Level 1）**：在传统强表格模型（含指数时间衰减特征 `up_order_strike`、共现特征与周期统计的 LightGBM）之上，引入商品行为向量（SVD / Word2Vec）或语义向量是否产生统计显著且超越实用门槛的增量？
2. **序列数学与损失函数正交归因（Level 2）**：序列深度学习模型（GRU、SASRec、eSASRec/LiGR-SASRec、HSTU-Math）相比传统模型或浅层池化的效果差异，究竟来自**训练目标与负采样规模（`BCE-1` vs `BCE-256` vs `Sampled Softmax-256`）**，还是来自**序列注意力数学结构（Softmax Attention vs LiGR 门控预归一化 vs HSTU Pointwise Aggregated Attention + 相对时间偏置）**？
3. **跨模型互补性（Hybrid）**：当把序列模型的打分 `sequence_logit` 与用户—商品表征余弦相似度 `embedding_similarity` 作为时间外（Out-of-Time / Out-of-Fold）特征并入强 LightGBM 时，能否突破单一模型范式的天花板？
4. **跨域业务结构调节效应（Dual-Domain）**：在**强周期复购主导的 B2B 快消补货场景（`swire_ro`）**与**长尾探索占比更高的 B2C 时尚服装场景（`H&M`）**之间，传统衰减特征与深度序列模型的相对收益如何随数据复购率（Repeat vs Exploration）及客户序列长度（S0/S1/S2）发生结构性迁移？
5. **计算架构工程前沿（Level 3）**：对经证实具有正向预测增量的序列模型，在固定数学结构下，`PyTorch Eager` vs `torch.compile` vs `AMP (BF16/FP16)` vs `Triton / SDPA Kernel` 的训练吞吐、P95 推理延迟、显存峰值与精度保持度如何权衡？

### 1.2 已锁定的三项顶层拍板决策
1. **数据集范围（双域对照）**：同时覆盖公开 **Kaggle H&M**（日级时尚服装，主外部基准）与本机 **`swire_ro`**（周级快消 SKU 补货，1.55GB 订单，仅本机运行不出网）。
2. **样本规模策略（分层渐进）**：
   - **R0（数据画像与契约）**：全量扫描 `H&M` 与 `swire_ro` 产出确定性五项画像与分位数。
   - **Tier-Smoke（2 万活跃客户分层抽样）**：用于跑通流水线、验证防泄漏门禁与模型前向/反向数值正确性。
   - **Tier-Dev（10 万客户按 S0/S1/S2 固定种子分层抽样）**：用于完成 R1（`B0–B2`）与 R2（`2×4 + 1` 神经序列正交矩阵 + `Hybrid`）的完整对比实验。
   - **Tier-Full（全量客户封存确认）**：在 W5 封存窗口上对胜出代表模型执行一次性终审评测。
3. **代码与治理边界**：在独立目录 `C:/Users/ghb/ZCodeProject/recsys-benchmark/` 下构建模块化代码库；只读复用 `C:/Users/ghb/ZCodeProject/swire_ro/`、`C:/Users/ghb/ZCodeProject/sku-replenishment-recommender/` 与 `C:/Users/ghb/ECC/cultivating-ml-agent/`；保持 `ExperimentIR 0.1.0` Schema 零修改，通过 `candidates[].params` 中的不可变协议文件路径与 `sha256` 摘要进行关联。

---

## 2. 系统总体架构与目录结构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     TopPrism RecSys Benchmark v0.9                       │
│               (C:/Users/ghb/ZCodeProject/recsys-benchmark)               │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│  1. Data & Split  │     │ 2. Eval & Metric  │     │ 3. Governance Bridge │
│  src/data/        │     │ src/eval/         │     │ src/governance/      │
│  - hm_adapter.py  │     │ - map12.py        │     │ - ir_bridge.py       │
│  - swire_adapter  │     │ - repeat_explore  │     │ - hash_lock.py       │
│  - window_split   │     │ - strata.py       │     │ (Links ExperimentIR  │
│  - baskets.py     │     │ - bootstrap_ci.py │     │  0.1.0 & Gate G1-G6) │
└─────────┬─────────┘     └─────────┬─────────┘     └──────────┬───────────┘
          │                         │                          │
          └─────────────────────────┼──────────────────────────┘
                                    │ Frozen Window Parquets + SHA256
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│  R1: Tabular &    │     │  R2: Sequential   │     │  R3: Compute &       │
│  Embedding Track  │     │  2x4+1 Matrix     │     │  Kernel Pareto       │
│  src/models/r1/   │     │  src/models/r2/   │     │  src/models/r3/      │
│  - candidates.py  │     │  - seq_dataset.py │     │  - bench_runner.py   │
│  - features.py    │     │  - losses.py      │     │  - eager / compile   │
│  - b0_rule.py     │     │  - encoders.py    │     │  - amp_bf16 / sdpa   │
│  - b1_lgbm.py     │     │    (GRU/SASRec/   │     │  - pareto_report.py  │
│  - b2_emb_lgbm.py │     │     LiGR/HSTU)    │     └──────────────────────┘
└─────────┬─────────┘     │  - hybrid_lgbm.py │
          │               └─────────┬─────────┘
          └────────────┬────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   Unified Evaluation &       │
        │   Stratified Report Engine   │
        │   artifacts/{r0,r1,r2,r3}/   │
        └──────────────────────────────┘
```

### 2.1 目录与文件职责清单

```
C:/Users/ghb/ZCodeProject/recsys-benchmark/
├── pyproject.toml                       # 依赖声明与 pytest 配置
├── protocols/
│   └── recsys-contract-v0.9.json        # 不可变研究协议配置（窗口日期、分层阈值、负采样数、随机种子）
├── docs/
│   └── superpowers/
│       ├── specs/
│       │   └── 2026-10-25-topprism-recsys-benchmark-v09-design.md
│       └── plans/
│           └── 2026-10-25-topprism-recsys-benchmark-r0-r3-plan.md
├── src/
│   ├── __init__.py
│   ├── contracts.py                     # 强类型数据结构与 Schema 校验器
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_hm.py               # 从 HuggingFace/Kaggle 拉取 H&M 三张核心表并校验 SHA256
│   │   ├── hm_adapter.py                # H&M 原始表转标准事件流 (customer_id, article_id, event_date, price)
│   │   ├── swire_adapter.py             # swire_ro 原始表转标准事件流 (OUTLET->customer_id, ARTICLE->article_id)
│   │   ├── window_builder.py            # W1-W5 时间切分、日级购物篮聚合、分层渐进抽样器 (20k/100k/Full)
│   │   └── profiler.py                  # R0 五项数据画像计算器（严守 W5 标签封存）
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py                   # 确定性平局打破的 MAP@12、Recall@K、Candidate Oracle、F1@K
│   │   ├── decomposition.py             # Repeat vs Exploration 严格可加性 AP 分解与 S0/S1/S2 分层聚合
│   │   └── stats.py                     # 客户级配对 Bootstrap 95% 置信区间与实用显著性检验
│   ├── governance/
│   │   ├── __init__.py
│   │   └── ir_bridge.py                 # 对接 cultivating-ml-agent 的 ExperimentIR 0.1.0 校验与门禁 G1-G6
│   └── models/
│       ├── __init__.py
│       ├── r1_candidates.py             # 多路候选召回器 (REPEAT, TIME_DECAY, POPULAR, ITEM_COOC, EMBEDDING)
│       ├── r1_features.py               # 零泄漏特征工程 (含 up_order_strike 日/周衰减、共现、周期统计、SVD/W2V)
│       ├── r1_baselines.py              # B0 (Rule)、B1 (LightGBM)、B2 (LightGBM + Behavior Emb)
│       ├── r2_losses.py                 # BCE-1、BCE-256、Sampled Softmax-256 (含 LogQ 流行度偏差校正开关)
│       ├── r2_encoders.py               # 同框架纯 PyTorch 编码器：Pool、GRU、SASRec、LiGR-SASRec、HSTU-Math
│       ├── r2_trainer.py                # 统一序列训练与 Track A (候选打分) / Track B (全库 MIPS) 推理器
│       ├── r2_hybrid.py                 # OOF 神经打分与余弦相似度注入 LightGBM 的 Hybrid 融合器
│       └── r3_compute_bench.py          # Eager vs torch.compile vs AMP (BF16/FP16) vs SDPA 吞吐/延迟/显存评测
└── tests/
    ├── test_r0_golden.py                # R0 强制要求的 12 项黄金单元测试
    ├── test_r1_pipeline.py              # R1 候选召回、特征防泄漏与 Oracle 不等式测试
    └── test_r2_orthogonal.py            # R2 损失函数、编码器形状/因果掩码一致性的单元测试
```

---

## 3. 核心数据契约与双域统一抽象（R0 规范）

### 3.1 双域标准事件流（Canonical Event Stream）
为使 H&M 与 `swire_ro` 共享同一套切分器、分层器、评价器与序列模型，所有原始输入首先映射为四列标准表 `events`（`parquet` 存储）：
- `customer_id: str` — 无损字符串编码（H&M 为 64 位十六进制字符串；`swire_ro` 为 `str(OUTLET).strip()`）。
- `article_id: str` — 无损字符串编码（H&M 强制 10 位前导零字符串 `zfill(10)`，防止 `0108775015` 丢前导零；`swire_ro` 为 `str(ARTICLE).strip()`）。
- `event_date: str (YYYY-MM-DD)` — 交易自然日（H&M 为 `t_dat`；`swire_ro` 为 `ORDER_DATE[:10]`，且仅保留 `SETTLE_CASE > 0` 的有效结算正单，过滤退货/取消单）。
- `value: float` — 交易金额或箱数（H&M 为 `price`；`swire_ro` 为 `SETTLE_CASE`）。

### 3.2 滚动时间窗口定义（W1–W5）
每个窗口由预测截止日 $t_{\text{cutoff}}$ 定义：
- **历史可见区间（History）**：$\text{event\_date} \le t_{\text{cutoff}}$。任何晚于 $t_{\text{cutoff}}$ 的记录在特征与候选生成阶段物理不可见。
- **标签预测区间（Label Horizon）**：$\text{event\_date} \in (t_{\text{cutoff}}, t_{\text{cutoff}} + \Delta]$，其中两域均固定为 **未来 7 个完整自然日（$\Delta = 7\text{ days}$）**。
- **H&M 五个标准窗口**：
  - `W1`：$t_{\text{cutoff}} = \text{2020-08-18}$，预测区间 `2020-08-19 .. 2020-08-25`（开发验证）
  - `W2`：$t_{\text{cutoff}} = \text{2020-08-25}$，预测区间 `2020-08-26 .. 2020-09-01`（开发验证）
  - `W3`：$t_{\text{cutoff}} = \text{2020-09-01}$，预测区间 `2020-09-02 .. 2020-09-08`（开发验证）
  - `W4`：$t_{\text{cutoff}} = \text{2020-09-08}$，预测区间 `2020-09-09 .. 2020-09-15`（开发验证）
  - `W5`：$t_{\text{cutoff}} = \text{2020-09-15}$，预测区间 `2020-09-16 .. 2020-09-22`（**封存测试窗口，R0/R1/R2 调参阶段严禁读取 W5 标签**）
- **`swire_ro` 对照窗口**：
  - 取数据集最末 5 个完整自然周作为 `SW_W1 .. SW_W4`（开发验证）与 `SW_W5`（封存确认），同样遵循 $t_{\text{cutoff}}$ 前一日截断与 7 天预测窗口。

### 3.3 日级购物篮聚合（Daily Basket Canonicalization）
由于 H&M 与 `swire_ro` 的原始 CSV 在同一客户同一天内的多行记录无毫秒级时间戳，禁止把原始 CSV 行顺序当作真实因果顺序：
- 同一 `(customer_id, event_date)` 内的购买商品去重后按 `article_id` 字典序升序排列，构成一个确定性日级购物篮 $B(u, d) = \{i_1 < i_2 < \dots < i_m\}$。
- 客户 $u$ 截至 $t_{\text{cutoff}}$ 的历史购物篮序列按日期升序排列：$S(u, t_{\text{cutoff}}) = [B(u, d_1), B(u, d_2), \dots, B(u, d_L)]$，其中 $L$ 为独立购物篮数（即有购买的独立自然日数）。

### 3.4 客户三层分层器（Customer Stratification）
仅使用截至 $t_{\text{cutoff}}$ 的历史信息，为每位客户计算：
- `historical_basket_count` ($L$)：截至 $t_{\text{cutoff}}$ 的独立购买自然日数。
- `days_since_last_purchase` ($D$)：$t_{\text{cutoff}} - d_L$（若 $L=0$ 则记为 $\infty / -1$）。
- **主分层（Stratum）**：
  - `S0`：$L \in \{0, 1\}$（冷启动或单次购买客户，无跨购物篮序列迁移信息）
  - `S1`：$L \in \{2, 3, 4\}$（短序列客户）
  - `S2`：$L \ge 5$（活跃多购物篮序列客户，检验高阶序列依赖的核心人群）
- **正交新鲜度子分层（Recency Bucket）**：
  - `Active`：$0 \le D \le 30$ 天
  - `Warm`：$31 \le D \le 90$ 天
  - `Dormant`：$D > 90$ 天或 $L = 0$

---

## 4. 评价数学与 Repeat / Exploration 严格可加性分解

### 4.1 确定性排序与 MAP@12
对于任一评估客户 $u \in U_{\text{eval}}$（在 Kaggle-compatible 口径下，$U_{\text{eval}} = \{u \mid |Y_u| > 0\}$，即测试窗口内实际发生购买的客户）：
1. 模型输出的候选打分表 `(article_id, score)` 先按 `article_id` 去重（保留最高分），再按 **`(-score, article_id)` 升序** 排序（即同分时严格按商品 ID 字典序打破平局，完全确定性），截取前 12 个商品：$R_u = [r_{u,1}, r_{u,2}, \dots, r_{u,12}]$。
2. 令真实购买集合为 $Y_u$（去重集合），分母常数 $M_u = \min(|Y_u|, 12)$。
3. 位置 $k \in \{1, \dots, 12\}$ 处的精度为 $P_u(k) = \frac{1}{k} \sum_{j=1}^{k} \mathbb{I}(r_{u,j} \in Y_u)$。
4. 客户 $u$ 的平均精度为：
   $$\text{AP@12}_u = \frac{1}{M_u} \sum_{k=1}^{12} P_u(k) \cdot \mathbb{I}(r_{u,k} \in Y_u)$$

### 4.2 Repeat vs Exploration 的严格可加性分解（Additive AP Decomposition）
设客户 $u$ 截至 $t_{\text{cutoff}}$ 的历史已购商品集合为 $H_u$。将测试窗口真实购买集合正交划分为：
$$Y_u^{\text{repeat}} = Y_u \cap H_u, \qquad Y_u^{\text{explore}} = Y_u \setminus H_u, \qquad Y_u^{\text{repeat}} \cap Y_u^{\text{explore}} = \emptyset$$
定义复购命中分量与探索命中分量（保持原始排名精度 $P_u(k)$ 与原始分母 $M_u$ 不变）：
$$\text{AP@12}_u^{\text{repeat}} = \frac{1}{M_u} \sum_{k=1}^{12} P_u(k) \cdot \mathbb{I}(r_{u,k} \in Y_u^{\text{repeat}})$$
$$\text{AP@12}_u^{\text{explore}} = \frac{1}{M_u} \sum_{k=1}^{12} P_u(k) \cdot \mathbb{I}(r_{u,k} \in Y_u^{\text{explore}})$$
**数学不变量**：对任意客户 $u$ 及全体客户均值，恒有：
$$\text{AP@12}_u^{\text{repeat}} + \text{AP@12}_u^{\text{explore}} \equiv \text{AP@12}_u, \qquad \text{MAP@12}_{\text{repeat}} + \text{MAP@12}_{\text{explore}} \equiv \text{MAP@12}$$

### 4.3 Candidate Oracle MAP@12（候选集信息上限）
对于大小不超过 $K$ 的候选集合 $C_u$，其在完美排序器下的理论上限为：将 $C_u \cap Y_u$ 中的所有命中商品全部排在最前面（第 $1, 2, \dots, |C_u \cap Y_u|$ 位）时的 $\text{AP@12}$：
$$\text{OracleAP@12}(C_u, Y_u) = \frac{\min(|C_u \cap Y_u|, 12)}{\min(|Y_u|, 12)}$$
**单调性不变量**：对基于候选集 $C_u$ 打分的任何排序模型 $M$，必有 $\text{MAP@12}(M) \le \text{OracleMAP@12}(C)$。

---

## 5. R1–R3 模型矩阵与实验规格

### 5.1 R1：传统与向量基线矩阵（固定候选池 $K$）
1. **多路召回器（`r1_candidates.py`）**：
   - `REPEAT`：客户历史已购商品，按指数衰减得分 $\sum_{b} 2^{-\Delta d_b / \tau}$ 排序（H&M 默认半衰期 $\tau = 7\text{ days}$，`swire_ro` 为 $\tau = 14\text{ days}$）。
   - `POPULAR`：截止日前 7 天 / 28 天全站及同年龄段（H&M `age` 分桶）或同渠道（`swire_ro` `CHANNEL_CODE`）热销商品。
   - `ITEM_COOC`：基于截止日前历史同购物篮共现矩阵 $W_{ij}$，由客户近 3 次购物篮商品触发的 Top 共现商品。
   - `EMBEDDING`：基于截止日前历史训练的 TruncatedSVD（32维）商品向量近邻召回。
2. **R1 排序模型对照（`r1_baselines.py`）**：
   - **`B0 (Rule Ensemble)`**：对 `REPEAT + ITEM_COOC + POPULAR` 的归一化分数做确定性加权融合。
   - **`B1 (Strong LightGBM)`**：输入客户统计特征、商品热度趋势、客户—商品交互特征（购买次数、距今天数、**`up_order_strike` 指数衰减分**、购买间隔均值/标准差、渠道/品类偏好比）、以及各召回源的 `source_score / source_rank`。
   - **`B2 (LightGBM + Behavior Embedding)`**：在 `B1` 特征基础上，严格追加由截止日前购买矩阵提取的 32 维 SVD + Word2Vec 行为表征特征（客户向量与候选商品向量的余弦相似度、点积及前 8 维分量）。

### 5.2 R2：纯 PyTorch 神经序列正交矩阵（`2×4 + 1`）与 Hybrid 融合
为彻底隔离“负样本数量”、“损失函数形式”与“序列注意力数学机制”，所有序列模型在**同一个纯 PyTorch 训练框架（`r2_trainer.py`）**下运行，共享完全相同的 Embedding 维度（$d = 128$）、最大序列长度（$L_{\max} = 50$ 个近期购物篮商品）、学习率调度与随机种子：

| 编码器架构 $\backslash$ 训练损失与负采样 | 1-neg BCE（历史锚点） | 256-neg BCE（独立 Sigmoid） | 256-neg Sampled Softmax（含 LogQ 校正） |
|---|---|---|---|
| **0. MeanPool（无位置/无注意力诊断）** | — | `Pool + BCE-256`（诊断） | `Pool + SS-256`（诊断） |
| **1. GRU（循环序列基线）** | — | **必跑 (`M_GRU_BCE256`)** | **必跑 (`M_GRU_SS256`)** |
| **2. SASRec（标准 Post-LN Softmax Attention）** | **历史锚点 (`M_SAS_BCE1`)** | **必跑 (`M_SAS_BCE256`)** | **必跑 (`M_SAS_SS256`)** |
| **3. LiGR-SASRec（Pre-LN + 门控残差，即 eSASRec 主干）** | — | **必跑 (`M_LIGR_BCE256`)** | **必跑 (`M_LIGR_SS256` = 完整 eSASRec)** |
| **4. HSTU-Math（Pointwise Aggregated Attention + 相对时间桶偏置 + 门控线性层）** | — | **必跑 (`M_HSTU_BCE256`)** | **必跑 (`M_HSTU_SS256`)** |

- **HSTU-Math 的精确数学定义（`r2_encoders.py`）**：
  每层将输入序列 $X \in \mathbb{R}^{B \times N \times d}$ 经线性投影并分拆为 $U, V, Q, K = \text{Split}(\phi(\text{Linear}(X)))$，计算**逐点聚合注意力（无 Softmax 归一化分母）**：
  $$A_{ij} = \frac{1}{N} \left(\text{SiLU}(Q_i K_j^\top + b_{\text{rel}}(t_i - t_j)) \odot M_{ij}^{\text{causal}}\right), \qquad Y = \text{Linear}\big(\text{LayerNorm}(A V) \odot U\big) + X$$
  并提供单元测试验证因果掩码零未来泄漏（修改 $t > i$ 的输入，位置 $i$ 的输出与梯度逐位不变）。
- **三种评测轨道**：
  1. **Track A（固定候选池重排）**：对 R1 冻结的同一候选池 $C_u$，用序列末端用户表征 $u_t$ 与候选商品向量 $e_i$ 计算 `sequence_logit = <u_t, e_i>` 排序。
  2. **Track B（独立全库检索）**：直接用 $u_t$ 对全商品库做矩阵内积检索 Top-12（对 `S0` 无历史客户回退到同年龄热门，以保持全集可比）。
  3. **Track C（Hybrid GBDT + Neural）**：将训练窗口 (W1–W3) 上的时间外神经打分 `sequence_logit` 与 `embedding_similarity` 并入 `B1 (LightGBM)`，在 W4（开发）与 W5（封存）上评测 `B1 + Neural` 是否显著优于 `B1 Alone` 与 `Neural Alone`。

### 5.3 R3：单卡 3080Ti 计算架构前沿评测
仅针对 R2 胜出的序列架构（如 `LiGR-SASRec` 与 `HSTU-Math`），固定模型超参，在本机 RTX 3080Ti (20GB, SM86) 上评测 4 种执行后端：
1. `PyTorch Eager (FP32)`
2. `PyTorch AMP (BF16)`
3. `torch.compile (inductor, BF16)`
4. `SDPA / Custom Triton Kernel (BF16)`
记录指标：`train_samples_per_sec`、`infer_users_per_sec`、`infer_p95_latency_ms`、`peak_vram_mb`、`top12_jaccard_vs_eager`。

---

## 6. 规格自检记录（Superpowers Inline Self-Review）

1. **占位符扫描**：全文无 `TODO`、`TBD`、或未定义的接口；双域字段名（H&M `customer_id/article_id/t_dat/price` 与 `swire_ro` `OUTLET/ARTICLE/ORDER_DATE/SETTLE_CASE`）均已核对真实文件表头。
2. **内部一致性**：R0 的三层客户分层（S0/S1/S2）、Repeat/Exploration 可加性分解公式、R1 的候选池接口、以及 R2 的 `2×4 + 1` 矩阵完全闭环对齐 GPT v0.9 裁决。
3. **范围检查**：R0–R3 四个模块职责单一、顺序递进，可在单个实现计划中拆分为 6 个独立可测的 TDD 任务。
4. **模糊性消除**：明确了同日多件订单按 `article_id` 字典序规范化为日级购物篮；明确了同分预测按 `article_id` 升序打破平局；明确了 `ExperimentIR 0.1.0` 通过 `candidates[].params` 挂载 `recsys_contract_path` 与 `recsys_contract_sha256`。
