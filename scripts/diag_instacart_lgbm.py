"""Reproduce the LGBM 0.0003 anomaly: load cached model, inspect features & score spread."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from run_instacart_lgbm import load_instacart, dense_cooc, feature_rows_block  # noqa: E402

import joblib  # noqa: E402


def main() -> None:
    clf = joblib.load(REPO / "artifacts/eval/instacart_cache/lgbm_model.joblib")
    seqs, num_items = load_instacart()
    users = sorted(seqs)
    pop = np.zeros(num_items + 1, dtype=np.float32)
    for u in users:
        for it in seqs[u][:-2]:
            pop[it] += 1.0

    probe = users[:3]
    all_ids = np.arange(1, num_items + 1)

    # (a) no-cooc features only check: build WITHOUT C to isolate (zeros for f2/f3)
    C_zero = np.zeros((num_items + 1, num_items + 1), dtype=np.float32)
    F = feature_rows_block([seqs[u][:-1] for u in probe], all_ids, C_zero, pop)
    print("feat-noCooc nonzero frac per col:", [round(float((F[:, :, k] != 0).mean()), 4) for k in range(5)])
    s = clf.predict_proba(F.reshape(-1, 5))[:, 1]
    print("scores(noCooc): std=%.3e uniq=%d min=%.3e max=%.3e" % (s.std(), len(np.unique(s)), s.min(), s.max()))
    for i, u in enumerate(probe):
        tgt = seqs[u][-1]
        rank = int((s[i * num_items:(i + 1) * num_items] >= s[i * num_items + tgt - 1]).sum())
        print(f"  user {u}: target {tgt} rank {rank}/{num_items}")

    # (b) with real cooc for the same users (slow: build small C from their histories only)
    # skip full C (6.8GB) — the zero-C probe already isolates whether model uses f0/f1/f4.
    print("DONE")


if __name__ == "__main__":
    main()
