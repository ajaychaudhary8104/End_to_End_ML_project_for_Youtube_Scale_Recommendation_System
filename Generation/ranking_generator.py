from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class RankingGenerator:
    """
    Phase 14 — Ranking Generator

    Creates ranking datasets with query-level candidate scoring
    suitable for XGBoost / LightGBM / CatBoost style training.

    Outputs:
    - query_id
    - item_id
    - rank_label
    """

    def __init__(
        self,
        foundation,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.users = users
        self.items = items
        self.affinity_df = affinity_df

    def _resolve_users(self, users: Optional[pd.DataFrame]) -> pd.DataFrame:
        if users is None:
            if self.users is None:
                raise ValueError("users must be provided")
            users = self.users

        working = users.copy()
        if "user_id" not in working.columns:
            working["user_id"] = np.arange(1, len(working) + 1)

        return working

    def _resolve_items(self, items: Optional[pd.DataFrame]) -> pd.DataFrame:
        if items is None:
            if self.items is None:
                items = pd.DataFrame({
                    "item_id": np.arange(1, self.config.n_items + 1)
                })
            else:
                items = self.items

        working = items.copy()
        if "item_id" not in working.columns:
            working["item_id"] = np.arange(1, len(working) + 1)

        return working

    def _resolve_affinity(self, affinity_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if affinity_df is None:
            if self.affinity_df is None:
                affinity_df = pd.DataFrame({
                    "user_id": self.rng.integers(1, self.config.n_users + 1, size=100),
                    "item_id": self.rng.integers(1, self.config.n_items + 1, size=100),
                    "candidate_generation_score": self.rng.random(100),
                    "ranking_label": self.rng.random(100)
                })
            else:
                affinity_df = self.affinity_df

        return affinity_df.copy()

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate the ranking dataset.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        affinity_df = self._resolve_affinity(affinity_df)

        rows = []
        query_counter = 1

        for _, user in users.iterrows():
            item_subset = items.sample(
                n=min(5, len(items)),
                replace=False,
                random_state=int(self.rng.integers(0, 1_000_000))
            )

            for _, item in item_subset.iterrows():
                rank_score = float(
                    affinity_df.loc[
                        (affinity_df["user_id"] == user["user_id"])
                        & (affinity_df["item_id"] == item["item_id"]),
                        "ranking_label"
                    ].mean()
                    if (
                        "ranking_label" in affinity_df.columns
                        and not affinity_df.loc[
                            (affinity_df["user_id"] == user["user_id"])
                            & (affinity_df["item_id"] == item["item_id"])
                        ].empty
                    )
                    else self.rng.random()
                )

                rows.append({
                    "query_id": f"q{query_counter:09d}",
                    "user_id": int(user["user_id"]),
                    "item_id": int(item["item_id"]),
                    "rank_label": rank_score,
                })

            query_counter += 1

        return pd.DataFrame(rows)


class Phase14RankingLayer:
    """
    Orchestrator-style wrapper for Phase 14.
    """

    def __init__(
        self,
        foundation,
        users: pd.DataFrame,
        items: pd.DataFrame,
        affinity_df: pd.DataFrame
    ):
        self.foundation = foundation
        self.users = users
        self.items = items
        self.affinity_df = affinity_df
        self.engine = RankingGenerator(
            foundation,
            users=users,
            items=items,
            affinity_df=affinity_df
        )

    def generate(self) -> pd.DataFrame:
        return self.engine.generate(
            users=self.users,
            items=self.items,
            affinity_df=self.affinity_df
        )


if __name__ == "__main__":
    from foundation import FoundationLayer, GeneratorConfig

    config = GeneratorConfig(
        n_users=100,
        n_items=10,
        embedding_dim=32,
        random_state=42
    )

    foundation = FoundationLayer(config)

    users = pd.DataFrame({"user_id": [1, 2, 3]})
    items = pd.DataFrame({"item_id": [1, 2, 3, 4, 5]})
    affinity_df = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [1, 2, 2, 3, 1],
        "ranking_label": [0.8, 0.6, 0.5, 0.9, 0.7]
    })

    generator = RankingGenerator(
        foundation,
        users=users,
        items=items,
        affinity_df=affinity_df
    )

    result = generator.generate(users, items, affinity_df)
    print(result.head())
