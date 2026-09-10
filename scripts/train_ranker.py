"""Fit LambdaRank on held-out candidates, with disjoint users for early stopping."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.api.inference import RecommendationEngine
from src.ranking.lambdarank import candidate_groups
from src.utils import load_config


def main():
    cfg = load_config()
    rc = cfg["ranking"]
    engine = RecommendationEngine(cfg, load_ranker=False)
    targets = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "val.parquet")
    users = np.array(sorted(targets.user_id.unique()))
    np.random.default_rng(cfg["seed"]).shuffle(users)
    users = users[:rc["max_queries"]]
    if len(users) < 2:
        raise ValueError("Need at least two validation users for ranker fit/early stopping")
    split = max(1, min(len(users)-1, int(len(users)*(1-rc["validation_fraction"]))))
    x, y, groups = candidate_groups(engine, targets[targets.user_id.isin(users[:split])])
    vx, vy, vg = candidate_groups(engine, targets[targets.user_id.isin(users[split:])])
    if not y.any() or not vy.any():
        raise ValueError("No retrieved positive labels in fit or validation queries; improve retrieval or increase max_queries")
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=rc["n_estimators"],
                          num_leaves=rc["num_leaves"], learning_rate=rc["lambdarank_learning_rate"],
                          random_state=cfg["seed"], n_jobs=4, verbosity=-1)
    model.fit(x, y, group=groups, eval_set=[(vx, vy)], eval_group=[vg],
              eval_at=cfg["evaluation"]["ranking_k"],
              callbacks=[lgb.early_stopping(rc["early_stopping_rounds"]), lgb.log_evaluation(10)])
    out = Path(cfg["artifacts_dir"])
    model.booster_.save_model(str(out / "ranker.txt"))
    (out / "ranker_training.json").write_text(json.dumps({
        "fit_queries": len(groups), "validation_queries": len(vg),
        "best_iteration": model.best_iteration_, "metrics": model.best_score_,
        "protocol": "validation interactions partitioned by user; train-only features; test untouched",
    }, indent=2))


if __name__ == "__main__":
    main()
