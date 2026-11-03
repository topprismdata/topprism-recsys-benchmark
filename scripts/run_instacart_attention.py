"""Instacart test of the attention hypothesis — retail-domain cross-check (§7).

PROTOCOL (frozen before running; identical to ML-1M v2 except data source):
- Data    : Instacart official-competition mirror, flattened SKU events in true order
            (order_number asc, add_to_cart_order asc), 5-core iterative filter.
- Sample  : 20,000 users sampled with seed 42 (NBR literature uses ~20k; LGBM memory bound).
- Task    : LOO (second-to-last = validation, last = test), full ranking over all items,
            NO seen-item masking (same disclosure as ML-1M v2).
- Model   : d128 / 2L / 4H / L200, SS256+logq, AdamW 2e-3, batch 256, 100 epochs w/
            validation-HR@10 early stopping (patience 10), seed 42.
Encoders: mean_pool / gru / sasrec / hstu_aligned  -> artifacts/eval/instacart_attention.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from src.models.r2_encoders import build_r2_encoder
from src.models.r2_losses import SampledSoftmax256Loss

SEED = 42
N_USERS = 20000
L = 200
D = 128
LAYERS = 2
HEADS = 4
NEG = 256
BATCH = 256
EPOCHS = 100
LR = 2e-3
ENCODERS = ["mean_pool", "sasrec"]  # v3 rerun with per-user records (GPT verification #2); gru/hstu v2 results stand
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = REPO / "artifacts/eval/instacart_attention.json"
SEQS_PATH = REPO / "data/instacart/seqs.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_instacart():
    raw: Dict[int, list] = {int(u): s for u, s in json.loads(SEQS_PATH.read_text(encoding="utf-8")).items()}
    raw = {u: s for u, s in raw.items() if len(s) >= 2}
    rng = np.random.default_rng(SEED)
    users_all = sorted(raw)
    pick = sorted(rng.choice(users_all, size=min(N_USERS, len(users_all)), replace=False).tolist())
    seqs = {u: raw[u] for u in pick}
    items = sorted({it for s in seqs.values() for it in s})
    id_of = {it: i + 1 for i, it in enumerate(items)}  # 0 = pad
    seqs = {u: [id_of[i] for i in s] for u, s in seqs.items()}
    lens = np.array([len(s) for s in seqs.values()])
    log(f"[data] users={len(seqs)} items={len(items)} events={int(lens.sum())} "
        f"len mean={lens.mean():.1f} median={np.median(lens):.0f} p10={np.percentile(lens,10):.0f} p90={np.percentile(lens,90):.0f}")
    return seqs, len(items), id_of


def make_inputs(seqs_lists, max_len):
    rows = []
    for s in seqs_lists:
        ids = s[-max_len:]
        r = np.zeros(max_len, dtype=np.int64)
        r[: len(ids)] = ids
        rows.append(r)
    return torch.from_numpy(np.stack(rows))


@torch.no_grad()
def eval_hr(model, item_w, inputs, targets, ks=(10, 50)):
    model.eval()
    hr = {k: 0 for k in ks}
    ndcg = {10: 0.0}
    n = 0
    dev = next(model.parameters()).device
    tgt_all = np.asarray(targets)
    for i in range(0, len(inputs), 512):
        ids = inputs[i:i + 512].to(dev)
        ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
        reps = model.encode_last_step(ids, timestamps=ts)
        scores = reps @ item_w.t()
        scores[:, 0] = -float("inf")
        top50 = torch.topk(scores, k=50, dim=1).indices.cpu().numpy()
        top10 = top50[:, :10]
        tgt_blk = tgt_all[i:i + len(top50)][:, None]
        hit10 = top10 == tgt_blk
        hit50 = top50 == tgt_blk
        h10 = hit10.any(axis=1)
        h50 = hit50.any(axis=1)
        ranks = np.where(h10, hit10.argmax(axis=1), 0)
        ndcg_add = np.where(h10, 1.0 / np.log2(ranks + 2), 0.0)
        hr[10] += int(h10.sum())
        hr[50] += int(h50.sum())
        ndcg[10] += float(ndcg_add.sum())
        n += len(top50)
    return {**{f"hr{k}": hr[k] / n for k in ks}, "ndcg10": ndcg[10] / n, "n": n}


def train_encoder(name, seqs, num_items):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dev = torch.device(DEVICE)
    model = build_r2_encoder(name, num_items=num_items, embedding_dim=D, max_seq_len=L,
                             num_layers=LAYERS, num_heads=HEADS, dropout=0.1).to(dev)
    criterion = SampledSoftmax256Loss(num_negatives=NEG, logq_correction=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    users = sorted(seqs)
    item_pop = np.zeros(num_items + 1, dtype=np.float64)
    for s in seqs.values():
        for it in s:
            item_pop[it] += 1
    logq = torch.tensor(np.log(item_pop / max(item_pop[1:].sum(), 1) + 1e-9), dtype=torch.float32, device=dev)

    train_users = [u for u in users if len(seqs[u]) >= 2]
    val_users = [u for u in users if len(seqs[u]) >= 3]
    train_seqs = {u: seqs[u][:-1] for u in train_users}
    val_inputs = make_inputs([seqs[u][:-2] for u in val_users], L)
    val_targets = np.array([seqs[u][-2] for u in val_users], dtype=np.int64)
    test_inputs = make_inputs([seqs[u][:-1] for u in train_users], L)
    test_targets = np.array([seqs[u][-1] for u in train_users], dtype=np.int64)

    rows = make_inputs([train_seqs[u] for u in train_users], L)
    n_rows = len(rows)
    gen = torch.Generator(device=dev)
    gen.manual_seed(SEED)
    item_w = model.item_embedding.weight
    best_val, best_state, bad = -1.0, None, 0
    t0 = time.time()
    epoch = 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_rows)
        losses = []
        for start in range(0, n_rows, BATCH):
            idx = perm[start:start + BATCH]
            b_inp = rows[idx].to(dev)
            pos_t = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(b_inp.size(0), -1)
            seq_repr = model(b_inp, timestamps=pos_t)
            b_tgt = torch.zeros_like(b_inp)
            b_tgt[:, :-1] = b_inp[:, 1:]  # next-item labels (v2 label-shift fix)
            flat_mask = b_tgt > 0
            flat_repr = seq_repr[flat_mask]
            flat_pos = b_tgt[flat_mask]
            neg = torch.randint(1, num_items + 1, (flat_repr.size(0), NEG), device=dev, generator=gen)
            loss = criterion(flat_repr, item_w, flat_pos, neg, item_log_probs=logq)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if (epoch + 1) % 5 == 0:
            v = eval_hr(model, item_w, val_inputs, val_targets)
            log(f"  [{name}] epoch {epoch+1} loss={np.mean(losses):.4f} val_hr10={v['hr10']:.4f}")
            if v["hr10"] > best_val:
                best_val = v["hr10"]
                best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 10:
                    log(f"  [{name}] early stop at epoch {epoch+1}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = eval_hr(model, model.item_embedding.weight, test_inputs, test_targets)
    test["seconds"] = round(time.time() - t0, 1)
    test["best_val_hr10"] = round(best_val, 4)
    test["epochs_run"] = epoch + 1
    # per-user test records for Repeat/New grouping (GPT verification #2); user order = train_users
    hits = []
    dev = next(model.parameters()).device
    with torch.no_grad():
        model.eval()
        for i in range(0, len(test_inputs), 512):
            ids = test_inputs[i:i + 512].to(dev)
            ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
            reps = model.encode_last_step(ids, timestamps=ts)
            scores = reps @ model.item_embedding.weight.t()
            scores[:, 0] = -float("inf")
            top10 = torch.topk(scores, k=10, dim=1).indices.cpu().numpy()
            tb = test_targets[i:i + len(top10), None]
            hits.extend((top10 == tb).any(axis=1).tolist())
    repeats = [seqs[u][-1] in seqs[u][:-1] for u in train_users]
    test["per_user"] = {"hit10": [int(h) for h in hits], "repeat_target": [int(r) for r in repeats]}
    hr_r = sum(h for h, r in zip(hits, repeats) if r) / max(sum(repeats), 1)
    hr_n = sum(h for h, r in zip(hits, repeats) if not r) / max(len(repeats) - sum(repeats), 1)
    test["repeat_new_summary"] = {
        "repeat_group": {"n": int(sum(repeats)), "hr10": round(hr_r, 4)},
        "new_group": {"n": int(len(repeats) - sum(repeats)), "hr10": round(hr_n, 4)},
    }
    return test


def main() -> None:
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    seqs, num_items, _id_of = load_instacart()
    out.setdefault("protocol", {
        "data": "instacart_official_mirror", "users_sampled": N_USERS, "sample_seed": SEED,
        "core": "5-core iterative", "task": "LOO full-ranking no-seen-masking",
        "model": f"d{D}/2L/4H/L{L}", "loss": "SS256+logq", "epochs": EPOCHS, "lr": LR,
    })
    results = out.setdefault("results", {})
    for name in ENCODERS:
        if name in results:
            log(f"[skip] {name} already done")
            continue
        log(f"[train] {name}")
        results[name] = train_encoder(name, seqs, num_items)
        log(f"[result] {name}: {json.dumps(results[name])}")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    main()
