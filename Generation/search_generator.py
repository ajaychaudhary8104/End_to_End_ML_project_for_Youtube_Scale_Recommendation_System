from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class SearchGenerator:
    """
    Phase 16 — Search Generator

    Creates simple search-style recommendation datasets with semantic query
    representations and an observed clicked item.

    Output schema includes:
    - query
    - query_embedding
    - clicked_item
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
                    "search_click_score": self.rng.random(100)
                })
            else:
                affinity_df = self.affinity_df

        return affinity_df.copy()

    def _build_query_text(self, user_id: int, item_id: int) -> str:
        """
        Construct a simple semantic query string from user and item metadata.
        """

        return f"query_user_{user_id}_item_{item_id}"

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate a search recommendation dataset.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        affinity_df = self._resolve_affinity(affinity_df)

        rows = []
        for _, user in users.iterrows():
            for item in items.head(3).itertuples(index=False):
                clicked_item = int(item.item_id)
                query_embedding = self.rng.normal(
                    loc=0.0,
                    scale=1.0,
                    size=self.config.embedding_dim
                )
                query_embedding = query_embedding.astype(np.float32)

                label = float(
                    affinity_df.loc[
                        (affinity_df["user_id"] == user["user_id"])
                        & (affinity_df["item_id"] == clicked_item),
                        "search_click_score"
                    ].mean()
                    if (
                        "search_click_score" in affinity_df.columns
                        and not affinity_df.loc[
                            (affinity_df["user_id"] == user["user_id"])
                            & (affinity_df["item_id"] == clicked_item)
                        ].empty
                    )
                    else self.rng.random()
                )

                rows.append({
                    "query": self._build_query_text(
                        int(user["user_id"]),
                        clicked_item
                    ),
                    "query_embedding": query_embedding,
                    "clicked_item": clicked_item,
                    "user_id": int(user["user_id"]),
                    "search_click_score": label,
                })

        return pd.DataFrame(rows)


class Phase16SearchLayer:
    """
    Orchestrator-style wrapper for Phase 16.
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
        self.engine = SearchGenerator(
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
        n_users=10,
        n_items=20,
        embedding_dim=8,
        random_state=42
    )

    foundation = FoundationLayer(config)

    users = pd.DataFrame({"user_id": [1, 2, 3]})
    items = pd.DataFrame({"item_id": [1, 2, 3, 4, 5]})
    affinity_df = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [1, 2, 2, 3, 1],
        "search_click_score": [0.9, 0.6, 0.7, 0.8, 0.5]
    })

    generator = SearchGenerator(
        foundation,
        users=users,
        items=items,
        affinity_df=affinity_df
    )

    result = generator.generate(users, items, affinity_df)
    print(result.head())
