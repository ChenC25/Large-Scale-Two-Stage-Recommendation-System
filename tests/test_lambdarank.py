import numpy as np
import pandas as pd
from src.ranking.lambdarank import RankingFeatures, candidate_groups


def test_train_only_features_and_missing_items():
    train = pd.DataFrame({"user_id": ["u", "u"], "item_id": ["a", "b"],
                          "rating": [5., 3.], "timestamp": [1, 2], "category": ["c", "c"]})
    features = RankingFeatures(train)
    result = features.build("u", ["a", "missing"], [.8, .2])
    assert result.item_rating_mean.tolist() == [5., 0.]
    assert result.category_match.tolist() == [1., 0.]
    assert np.isfinite(result.to_numpy()).all()


def test_groups_keep_retrieval_misses_without_injection():
    class Engine:
        features = RankingFeatures(pd.DataFrame({"user_id": ["u"], "item_id": ["a"],
            "rating": [5.], "timestamp": [1]}))
        def candidates(self, user):
            return ["a", "b"], [0.8, 0.2]
    targets = pd.DataFrame({"user_id": ["u", "v"], "item_id": ["b", "z"], "label": [1, 1]})
    x, y, groups = candidate_groups(Engine(), targets)
    assert groups == [2, 2]
    assert y.tolist() == [0, 1, 0, 0]
    assert len(x) == sum(groups)


def test_faiss_padding_is_not_last_item():
    from src.retrieval.index import FAISSIndex
    class Index:
        def search(self, vectors, k):
            return np.array([[1., -np.inf]]), np.array([[0, -1]])
    index = FAISSIndex()
    index.index = Index()
    index.item_ids = np.array([42])
    _, ids = index.search(np.ones((1, 2)), 2)
    assert ids.tolist() == [[42, -1]]


def test_native_model_roundtrip(tmp_path):
    import lightgbm as lgb
    x = pd.DataFrame({name: np.tile([0., 1., 2.], 8)
                      for name in RankingFeatures.columns})
    y = np.tile([0, 0, 1], 8)
    model = lgb.LGBMRanker(n_estimators=5, min_child_samples=1,
                          num_leaves=3, n_jobs=1, verbosity=-1)
    model.fit(x, y, group=[3]*8)
    path = tmp_path / "ranker.txt"
    model.booster_.save_model(str(path))
    restored = lgb.Booster(model_file=str(path))
    scores = restored.predict(x, num_threads=1)
    assert np.allclose(scores, model.predict(x))
    assert scores[2] > scores[0]


def test_engine_serves_native_ranker(tmp_path):
    import json
    import torch
    import lightgbm as lgb
    from src.api.inference import RecommendationEngine
    from src.retrieval.two_tower import TwoTowerModel
    from src.utils import load_config
    cfg = load_config()
    cfg["data"]["processed_dir"] = str(tmp_path)
    cfg["artifacts_dir"] = str(tmp_path)
    cfg["index"]["index_type"] = "FlatIP"
    cfg["index"]["top_k"] = 5
    vocab = {"user_id": {"u": 1}, "item_id": {"a": 1, "b": 2}, "category": {"c": 1}}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    pd.DataFrame({"user_id": ["u", "u"], "item_id": ["a", "b"],
                  "rating": [5., 4.], "timestamp": [1, 2], "category": ["c", "c"]}).to_parquet(tmp_path / "train.parquet")
    rc = cfg["retrieval"]
    model = TwoTowerModel(num_users=2, num_items=3, num_categories=2,
        embedding_dim=rc["embedding_dim"], user_mlp_dims=rc["mlp_dims"],
        item_mlp_dims=[rc["embedding_dim"]], temperature=rc["temperature"])
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "retriever_best.pt")
    x = pd.DataFrame({name: [0., 1., 2., 3.] for name in RankingFeatures.columns})
    ranker = lgb.LGBMRanker(n_estimators=2, min_child_samples=1, n_jobs=1, verbosity=-1)
    ranker.fit(x, [0, 1, 0, 1], group=[2, 2])
    ranker.booster_.save_model(str(tmp_path / "ranker.txt"))
    engine = RecommendationEngine(cfg)
    response = engine.recommend("u", 20)
    assert {r["item_id"] for r in response["recommendations"]} == {"a", "b"}
    assert len(response["recommendations"]) == 2
    assert response["total_latency_ms"] >= 0
