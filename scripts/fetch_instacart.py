"""Fetch Instacart data via Kaggle API (token provided by user).

Primary: official competition instacart-market-basket-analysis (full: 3.4M orders, 206k users, 50k products).
Fallback: public dataset mirrors if competition rules not accepted.
Target: data/instacart/raw/
"""
import os
import sys

if not os.environ.get("KAGGLE_API_TOKEN"):
    sys.exit("Set KAGGLE_API_TOKEN in the environment before fetching Instacart data.")

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "instacart", "raw")
os.makedirs(RAW, exist_ok=True)

CANDIDATES = [
    ("competition", "instacart-market-basket-analysis"),
    ("dataset", "psparks/instacart-market-basket-analysis"),
    ("dataset", "suksma/the-instacart-online-grocery-basket-analysis-dataset"),
]

for kind, slug in CANDIDATES:
    try:
        import kagglehub
        if kind == "competition":
            path = kagglehub.competition_download(slug)
        else:
            path = kagglehub.dataset_download(slug)
        print(f"OK {kind}:{slug} -> {path}", flush=True)
        print("PATH:" + path)
        sys.exit(0)
    except Exception as e:
        print(f"FAIL {kind}:{slug}: {type(e).__name__}: {str(e)[:300]}", flush=True)

print("ALL_FAILED")
