"""Final dual-track analysis per PREREG v1.2 §3 (ITT main + conditional auxiliary)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "artifacts/eval/ml1m_ablation"
SEEDS = [42, 43, 44, 45, 46, 47]
COLLAPSE_VAL = 0.05  # frozen in v1.2
BOOT_N = 10000

R = json.loads((DIR / "results.json").read_text(encoding="utf-8"))
ARMS = ["sasrec_full", "sasrec_uniform", "sasrec_pos_only", "mean_pool"]
LABEL = dict(sasrec_full="A0", sasrec_uniform="A1", sasrec_pos_only="A2", mean_pool="A3")


def hits(arm, seed):
    return np.load(DIR / f"hits_{arm}_s{seed}.npz")["hits"].astype(np.float64)


def boot_ci(d, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(BOOT_N, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


collapsed = {a: [s for s in SEEDS if R[f"{a}_s{s}"]["best_val_hr10"] < COLLAPSE_VAL] for a in ARMS}
out = {
    "collapse_rule": f"best_val_hr10 < {COLLAPSE_VAL} (frozen v1.2)",
    "collapsed_arms": {LABEL[a]: collapsed[a] for a in ARMS},
    "train_success_rate": {LABEL[a]: f"{6 - len(collapsed[a])}/6" for a in ARMS},
}

# per-arm table
tbl = {}
for a in ARMS:
    per = {s: round(R[f"{a}_s{s}"]["hr10"], 4) for s in SEEDS}
    ok = [R[f"{a}_s{s}"]["hr10"] for s in SEEDS if s not in collapsed[a]]
    tbl[LABEL[a]] = {
        "per_seed_hr10": per,
        "itt_mean_all6": round(float(np.mean(list(per.values()))), 4),
        "cond_mean_success_only": round(float(np.mean(ok)), 4) if ok else None,
        "n_success": len(ok),
    }
out["arm_table"] = tbl

# conditional paired deltas (both arms non-collapsed)
def cond_delta(a, b):
    seeds = [s for s in SEEDS if s not in collapsed[a] and s not in collapsed[b]]
    res = {}
    for s in seeds:
        m, lo, hi = boot_ci(hits(a, s) - hits(b, s), s)
        res[f"s{s}"] = {"mean": round(m, 5), "ci95": [round(lo, 5), round(hi, 5)], "sig": bool(lo > 0)}
    vals = [v["mean"] for v in res.values()]
    return {"n_seeds": len(seeds), "per_seed": res,
            "mean": round(float(np.mean(vals)), 5) if vals else None,
            "all_point_estimates_positive": bool(all(v > 0 for v in vals)) if vals else None,
            "n_significant": sum(1 for v in res.values() if v["sig"])}

out["conditional"] = {
    "d1_A0_minus_A2": cond_delta("sasrec_full", "sasrec_pos_only"),
    "d2_A0_minus_A1": cond_delta("sasrec_full", "sasrec_uniform"),
    "h2a_A1_minus_A3": cond_delta("sasrec_uniform", "mean_pool"),
    "h2b_A2_minus_A3": cond_delta("sasrec_pos_only", "mean_pool"),
    "extra_pos_vs_uniform_A2_minus_A1_success_only": cond_delta("sasrec_pos_only", "sasrec_uniform"),
}

# ITT paired deltas (all seeds, failures count as results)
itt = {}
for tag, (a, b) in {"d1": ("sasrec_full", "sasrec_pos_only"), "d2": ("sasrec_full", "sasrec_uniform"),
                    "h2a": ("sasrec_uniform", "mean_pool"), "h2b": ("sasrec_pos_only", "mean_pool")}.items():
    per = {}
    for s in SEEDS:
        m, lo, hi = boot_ci(hits(a, s) - hits(b, s), s)
        per[f"s{s}"] = [round(m, 5), round(lo, 5), round(hi, 5)]
    vals = [v[0] for v in per.values()]
    itt[tag] = {"per_seed_mean_ci": per,
                "itt_mean": round(float(np.mean(vals)), 5),
                "min_seed": round(min(vals), 5), "max_seed": round(max(vals), 5)}
out["itt_main_analysis"] = itt

(DIR / "final_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
