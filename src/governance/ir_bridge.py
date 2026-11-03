import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict

CULTIVATING_ML_AGENT_SRC = Path("C:/Users/ghb/ECC/cultivating-ml-agent/framework/src")
if str(CULTIVATING_ML_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(CULTIVATING_ML_AGENT_SRC))

from ir.experiment_ir import authorize_execution, compute_content_hash, load_ir  # noqa: E402


def build_recsys_experiment_ir(contract_path: Path) -> Dict[str, Any]:
    raw_bytes = contract_path.read_bytes()
    contract_sha = hashlib.sha256(raw_bytes).hexdigest()
    protocol = {"strategy": "time_based", "n_folds": 4, "time_col": "event_date", "val_size_weeks": 1}
    ir_doc: Dict[str, Any] = {
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
        "expected_artifacts": [
            "artifacts/r0/profile_summary.json",
            "artifacts/r1/r1_summary.json",
            "artifacts/r2/r2_summary.json",
            "artifacts/r3/r3_summary.json",
        ],
        "authorization": {
            "authorized_by": "user_and_gpt_v09_signoff",
            "authorization_ref": "https://chatgpt.com/c/6adc27a7-7784-83ee-a6a1-6ef93d31d2a2#turn-12",
            "granted_at": "2026-10-25T11:15:00Z",
        },
        "source_ref": "protocols/recsys-contract-v0.9.json",
        "created_at": "2026-10-25T11:15:00Z",
        "revision": 1,
    }
    ir_doc["content_hash"] = compute_content_hash(ir_doc)
    return ir_doc


def validate_ir_with_gates(ir_path: Path) -> Dict[str, Any]:
    ir = load_ir(ir_path)
    verdict = authorize_execution(ir)
    return {
        "passed": verdict.ok,
        "content_hash": ir["content_hash"],
        "failed_gates": list(verdict.failed_gates),
    }
