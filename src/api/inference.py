import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from src.ranking.lambdarank import RankingFeatures

from src.retrieval.two_tower import TwoTowerModel
from src.retrieval.index import FAISSIndex
from src.data.vocab import load_vocab
from src.utils import load_config, vocab_sizes, build_item_category_arrays


class RecommendationEngine:
    def __init__(self, cfg, load_ranker=True):
        device = torch.device("cpu")
        self.device = device

        processed_dir = Path(cfg["data"]["processed_dir"])
        artifacts_dir = Path(cfg["artifacts_dir"])

        # load vocab and data
        self.vocab = load_vocab(str(processed_dir / "vocab.json"))
        train_df = pd.read_parquet(processed_dir / "train.parquet")
        num_users, num_items, num_categories = vocab_sizes(self.vocab)

        meta_path = processed_dir / "meta.parquet"
        meta_df = pd.read_parquet(meta_path) if meta_path.exists() else None

        # reverse vocab
        self.id_to_raw_item = {v: k for k, v in self.vocab["item_id"].items()}

        # load user tower from TwoTowerModel checkpoint
        ret_cfg = cfg["retrieval"]
        retrieval_model = TwoTowerModel(
            num_users=num_users, num_items=num_items, num_categories=num_categories,
            embedding_dim=ret_cfg["embedding_dim"],
            user_mlp_dims=ret_cfg["mlp_dims"],
            item_mlp_dims=[ret_cfg["embedding_dim"]],
            temperature=ret_cfg["temperature"],
        )
        ckpt = torch.load(artifacts_dir / "retriever_best.pt", map_location=device, weights_only=True)
        retrieval_model.load_state_dict(ckpt["model_state_dict"])
        retrieval_model.eval()
        self.user_tower = retrieval_model.user_tower

        # build user history lookup from training data
        self.max_history_length = cfg["data"]["max_history_length"]
        self.user_histories = {}
        sorted_df = train_df.sort_values("timestamp")
        for user_raw, group in sorted_df.groupby("user_id"):
            uid = self.vocab["user_id"].get(str(user_raw), 0)
            items = [self.vocab["item_id"].get(str(i), 0) for i in group["item_id"]]
            self.user_histories[uid] = items

        # load FAISS index
        idx_cfg = cfg["index"]
        self.retrieval_top_k = idx_cfg["top_k"]
        self.faiss_index = FAISSIndex(
            embedding_dim=ret_cfg["embedding_dim"],
            index_type=idx_cfg["index_type"],
            nprobe=idx_cfg["nprobe"],
        )
        index_path = str(artifacts_dir / "faiss.index")
        if Path(index_path).exists():
            self.faiss_index.load(index_path)
        else:
            all_item_ids, all_category_ids = build_item_category_arrays(train_df, self.vocab)
            self.faiss_index.build(
                item_tower=retrieval_model.item_tower,
                all_item_ids=all_item_ids,
                all_category_ids=all_category_ids,
                device=device,
                save_path=index_path,
            )

        self.features = RankingFeatures(train_df, meta_df)
        self.ranker = (lgb.Booster(model_file=str(artifacts_dir / "ranker.txt"))
                       if load_ranker else None)

        self.known_users = set(self.vocab["user_id"].keys())

    def _encode_user(self, user_id):
        uid = self.vocab["user_id"].get(user_id, 0)
        history = self.user_histories.get(uid, [])
        history = history[-self.max_history_length:]
        pad_len = self.max_history_length - len(history)
        history_padded = [0] * pad_len + history

        with torch.no_grad():
            uid_t = torch.tensor([uid], dtype=torch.long)
            hist_t = torch.tensor([history_padded], dtype=torch.long)
            emb = self.user_tower(uid_t, hist_t).numpy()
        return emb.astype(np.float32)

    def candidates(self, user_id):
        distances, ids = self.faiss_index.search(
            self._encode_user(user_id), self.retrieval_top_k)
        valid = ids[0] > 0
        raw = [self.id_to_raw_item[int(i)] for i in ids[0][valid]]
        return raw, distances[0][valid]

    def recommend(self, user_id, top_k=20):
        if self.ranker is None:
            raise RuntimeError("Ranker has not been loaded")
        start = time.perf_counter()
        candidates, retrieval_scores = self.candidates(user_id)
        retrieval_ms = (time.perf_counter() - start) * 1000
        t0 = time.perf_counter()
        scores = self.ranker.predict(self.features.build(user_id, candidates, retrieval_scores), num_threads=1)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return {
            "user_id": user_id,
            "recommendations": [
                {"item_id": candidates[i], "score": float(scores[i]), "rank": rank}
                for rank, i in enumerate(order, 1)],
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "ranking_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "total_latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
