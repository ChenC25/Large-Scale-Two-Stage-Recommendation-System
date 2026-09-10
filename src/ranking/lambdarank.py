"""Shared, train-only tabular features for LambdaRank training and serving."""
import numpy as np
import pandas as pd
from src.utils import build_user_cat_stats
from src.data.features import build_user_features


class RankingFeatures:
    columns = ["retrieval_score", "user_rating_mean", "user_interactions",
               "item_rating_mean", "item_popularity", "category_match",
               "user_category_rating_mean"]

    def __init__(self, train_df, meta_df=None):
        self.users = build_user_features(train_df).to_dict("index")
        self.items = train_df.groupby("item_id")["rating"].agg(["mean", "count"]).to_dict("index")
        source = train_df if "category" in train_df else meta_df
        self.categories = (source.drop_duplicates("item_id").set_index("item_id")["category"].to_dict()
                           if source is not None else {})
        self.cross = build_user_cat_stats(train_df, meta_df)

    def build(self, user_id, item_ids, retrieval_scores):
        if len(item_ids) != len(retrieval_scores):
            raise ValueError("Candidate IDs and scores must align")
        user = self.users.get(user_id, {})
        rows = []
        for item_id, score in zip(item_ids, retrieval_scores):
            item = self.items.get(item_id, {})
            cross = self.cross.get((user_id, self.categories.get(item_id, "")))
            rows.append([score, user.get("avg_rating", 0), user.get("num_interactions", 0),
                         item.get("mean", 0), item.get("count", 0),
                         float(cross is not None), cross if cross is not None else 0])
        return pd.DataFrame(rows, columns=self.columns, dtype=np.float32).replace(
            [np.inf, -np.inf], np.nan).fillna(0)


def candidate_groups(engine, targets):
    """One contiguous query per user; never inject held-out positives into candidates."""
    frames, labels, groups = [], [], []
    for user, group in targets.groupby("user_id", sort=True):
        ids, scores = engine.candidates(str(user))
        if not ids:
            continue
        relevant = set(group.loc[group["label"] == 1, "item_id"].astype(str))
        frames.append(engine.features.build(str(user), ids, scores))
        labels.extend(int(item in relevant) for item in ids)
        groups.append(len(ids))
    if not frames:
        raise ValueError("No ranking queries; preprocess data and train retrieval first")
    return pd.concat(frames, ignore_index=True), np.asarray(labels), groups
