from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class GraphRecommendationBuilder:
    """
    Phase 21 — Graph Recommendation Builder

    Creates a lightweight LightGCN-style graph dataset with:
    - adjacency_matrix
    - edge_index
    - graph_features
    """

    def __init__(
        self,
        foundation,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        interactions: Optional[pd.DataFrame] = None,
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.users = users
        self.items = items
        self.interactions = interactions

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
                    "item_id": np.arange(1, self.config.n_items + 1),
                    "genre": self.rng.choice(["Action", "Drama", "Comedy"], size=self.config.n_items),
                    "popularity_score": self.rng.random(self.config.n_items),
                })
            else:
                items = self.items

        working = items.copy()
        if "item_id" not in working.columns:
            working["item_id"] = np.arange(1, len(working) + 1)

        return working

    def _resolve_interactions(self, interactions: Optional[pd.DataFrame]) -> pd.DataFrame:
        if interactions is None:
            if self.interactions is None:
                interactions = pd.DataFrame({
                    "user_id": self.rng.integers(1, self.config.n_users + 1, size=50),
                    "item_id": self.rng.integers(1, self.config.n_items + 1, size=50),
                    "interaction_weight": self.rng.random(50),
                })
            else:
                interactions = self.interactions

        working = interactions.copy()
        if "user_id" not in working.columns:
            working["user_id"] = self.rng.integers(1, self.config.n_users + 1, size=len(working))
        if "item_id" not in working.columns:
            working["item_id"] = self.rng.integers(1, self.config.n_items + 1, size=len(working))
        if "interaction_weight" not in working.columns:
            working["interaction_weight"] = self.rng.random(len(working))

        return working

    def build_adjacency_matrix(self, n_users: int, n_items: int, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Build a simple user-item adjacency matrix.
        """

        matrix = np.zeros((n_users, n_items), dtype=np.float32)
        for _, row in interactions.iterrows():
            user_idx = int(row["user_id"]) - 1
            item_idx = int(row["item_id"]) - 1
            matrix[user_idx, item_idx] = float(row["interaction_weight"])

        return pd.DataFrame(matrix)

    def build_edge_index(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Return the edge list in graph indexing form.
        """

        return interactions[["user_id", "item_id", "interaction_weight"]].copy()

    def build_graph_features(self, users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        """
        Build graph-level feature summaries for nodes.
        """

        rows = []
        for _, user in users.iterrows():
            rows.append({
                "node_id": int(user["user_id"]),
                "node_type": "user",
                "feature_scalar": float(user.get("engagement_score", self.rng.random())),
            })

        for _, item in items.iterrows():
            rows.append({
                "node_id": int(item["item_id"]),
                "node_type": "item",
                "feature_scalar": float(item.get("popularity_score", self.rng.random())),
            })

        return pd.DataFrame(rows)

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        interactions: Optional[pd.DataFrame] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate a LightGCN-style graph dataset.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        interactions = self._resolve_interactions(interactions)

        adjacency = self.build_adjacency_matrix(
            n_users=len(users),
            n_items=len(items),
            interactions=interactions,
        )
        edge_index = self.build_edge_index(interactions)
        graph_features = self.build_graph_features(users, items)

        return {
            "adjacency_matrix": adjacency,
            "edge_index": edge_index,
            "graph_features": graph_features,
        }


class Phase21GraphRecommendationLayer:
    """
    Orchestrator-style wrapper for Phase 21.
    """

    def __init__(
        self,
        foundation,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: pd.DataFrame,
    ):
        self.foundation = foundation
        self.users = users
        self.items = items
        self.interactions = interactions
        self.engine = GraphRecommendationBuilder(
            foundation,
            users=users,
            items=items,
            interactions=interactions,
        )

    def generate(self) -> dict[str, pd.DataFrame]:
        return self.engine.generate(
            users=self.users,
            items=self.items,
            interactions=self.interactions,
        )


if __name__ == "__main__":
    from foundation import FoundationLayer, GeneratorConfig

    config = GeneratorConfig(
        n_users=3,
        n_items=4,
        embedding_dim=8,
        random_state=42,
    )

    foundation = FoundationLayer(config)

    users = pd.DataFrame({
        "user_id": [1, 2, 3],
        "engagement_score": [0.9, 0.7, 0.5],
    })
    items = pd.DataFrame({
        "item_id": [1, 2, 3, 4],
        "popularity_score": [0.8, 0.6, 0.4, 0.2],
    })
    interactions = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [1, 2, 3],
        "interaction_weight": [0.9, 0.8, 0.7],
    })

    builder = GraphRecommendationBuilder(
        foundation,
        users=users,
        items=items,
        interactions=interactions,
    )

    result = builder.generate(users, items, interactions)
    print(result["adjacency_matrix"].head())
    print(result["edge_index"].head())
    print(result["graph_features"].head())
