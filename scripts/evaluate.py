"""Compare retrieval order with LambdaRank on untouched positive test targets."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.api.inference import RecommendationEngine
from src.eval.metrics import recall_at_k, ndcg_at_k, hit_rate_at_k, mrr
from src.utils import load_config


def main():
    cfg = load_config()
    engine = RecommendationEngine(cfg)
    test = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "test.parquet")
    test = test[test.label == 1]
    values = {}
    def add(key, value):
        values.setdefault(key, []).append(value)
    for user, group in tqdm(test.groupby("user_id"), desc="Test queries"):
        gt = group.item_id.astype(str).tolist()
        candidates, scores = engine.candidates(str(user))
        for k in cfg["evaluation"]["retrieval_k"]:
            if k > engine.retrieval_top_k:
                raise ValueError("index.top_k must cover all evaluation retrieval cutoffs")
            add(f"retrieval/recall@{k}", recall_at_k(gt, candidates, k))
        predicted = engine.ranker.predict(engine.features.build(str(user), candidates, scores), num_threads=1)
        ranked = [candidates[i] for i in np.argsort(-predicted, kind="stable")]
        for name, items in [("retrieval", candidates), ("lambdarank", ranked)]:
            for k in cfg["evaluation"]["ranking_k"]:
                add(f"{name}/ndcg@{k}", ndcg_at_k(gt, items, k))
                add(f"{name}/hit_rate@{k}", hit_rate_at_k(gt, items, k))
            add(f"{name}/mrr", mrr(gt, items))
    if not values:
        raise ValueError("No positive test queries")
    result = {key: float(np.mean(v)) for key, v in values.items()}
    result["queries"] = int(test.user_id.nunique())
    path = Path(cfg["artifacts_dir"]) / "evaluation_results.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
