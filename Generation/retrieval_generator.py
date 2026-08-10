from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class RetrievalGenerator:
    """
    Phase 13 — Retrieval Generator

    Builds retrieval-style training datasets.

    Supports:
    - two-tower retrieval
    - ANN retrieval style candidate scoring

    Output schema includes:
    - user_features
    - item_features
    - retrieval_label
    - query_embedding
    - candidate_embedding
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
                    "candidate_generation_score": self.rng.random(100)
                })
            else:
                affinity_df = self.affinity_df

        return affinity_df.copy()

    def generate_two_tower_dataset(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        affinity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create a two-tower-style retrieval dataset.
        """

        rows = []

        for _, user in users.iterrows():
            for _, item in items.head(5).iterrows():
                label = float(
                    affinity_df.loc[
                        (affinity_df["user_id"] == user["user_id"])
                        & (affinity_df["item_id"] == item["item_id"]),
                        "candidate_generation_score"
                    ].mean()
                    if (
                        "candidate_generation_score" in affinity_df.columns
                        and
                        not affinity_df.loc[
                            (affinity_df["user_id"] == user["user_id"])
                            & (affinity_df["item_id"] == item["item_id"])
                        ].empty
                    )
                    else self.rng.random()
                )

                rows.append({
                    "user_id": int(user["user_id"]),
                    "item_id": int(item["item_id"]),
                    "user_features": f"u{int(user['user_id'])}",
                    "item_features": f"i{int(item['item_id'])}",
                    "retrieval_label": label,
                })

        return pd.DataFrame(rows)

    def generate_ann_dataset(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        affinity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create a simple ANN retrieval dataset with embeddings.
        """

        rows = []
        for _, user in users.iterrows():
            for _, item in items.head(3).iterrows():
                query_embedding = np.array([
                    float(self.rng.random()),
                    float(self.rng.random()),
                    float(self.rng.random())
                ])
                candidate_embedding = np.array([
                    float(self.rng.random()),
                    float(self.rng.random()),
                    float(self.rng.random())
                ])

                rows.append({
                    "user_id": int(user["user_id"]),
                    "item_id": int(item["item_id"]),
                    "query_embedding": query_embedding,
                    "candidate_embedding": candidate_embedding,
                    "retrieval_label": float(
                        self.rng.random()
                    )
                })

        return pd.DataFrame(rows)

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Return both retrieval-style datasets.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        affinity_df = self._resolve_affinity(affinity_df)

        two_tower = self.generate_two_tower_dataset(users, items, affinity_df)
        ann = self.generate_ann_dataset(users, items, affinity_df)

        return {
            "two_tower_dataset": two_tower,
            "candidate_dataset": ann,
        }


class Phase13RetrievalLayer:
    """
    Orchestrator-style wrapper for Phase 13.
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
        self.engine = RetrievalGenerator(
            foundation,
            users=users,
            items=items,
            affinity_df=affinity_df
        )

    def generate(self) -> dict[str, pd.DataFrame]:
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
        "candidate_generation_score": [0.8, 0.6, 0.5, 0.9, 0.7]
    })

    generator = RetrievalGenerator(
        foundation,
        users=users,
        items=items,
        affinity_df=affinity_df
    )

    result = generator.generate(users, items, affinity_df)
    print(result["two_tower_dataset"].head())
    print(result["candidate_dataset"].head())
