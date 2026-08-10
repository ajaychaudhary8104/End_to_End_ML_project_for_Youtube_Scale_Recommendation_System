from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class FeatureStoreBuilder:
    """
    Phase 19 — Feature Store Builder

    Builds production-style feature store tables for:
    - user_features
    - item_features
    - context_features
    - affinity_features
    """

    def __init__(
        self,
        foundation,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        contexts: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None,
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.users = users
        self.items = items
        self.contexts = contexts
        self.affinity_df = affinity_df

    def _resolve_users(self, users: Optional[pd.DataFrame]) -> pd.DataFrame:
        if users is None:
            if self.users is None:
                users = pd.DataFrame({
                    "user_id": np.arange(1, self.config.n_users + 1),
                    "age": self.rng.integers(18, 70, size=self.config.n_users),
                    "engagement_score": self.rng.random(self.config.n_users),
                })
            else:
                users = self.users

        working = users.copy()
        if "user_id" not in working.columns:
            working["user_id"] = np.arange(1, len(working) + 1)
        if "event_timestamp" not in working.columns:
            working["event_timestamp"] = pd.Timestamp.utcnow()
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
        if "event_timestamp" not in working.columns:
            working["event_timestamp"] = pd.Timestamp.utcnow()
        return working

    def _resolve_contexts(self, contexts: Optional[pd.DataFrame]) -> pd.DataFrame:
        if contexts is None:
            if self.contexts is None:
                contexts = pd.DataFrame({
                    "context_id": np.arange(1, 20 + 1),
                    "watch_intent_score": self.rng.random(20),
                    "homepage_bias": self.rng.random(20),
                    "search_bias": self.rng.random(20),
                })
            else:
                contexts = self.contexts

        working = contexts.copy()
        if "context_id" not in working.columns:
            working["context_id"] = np.arange(1, len(working) + 1)
        if "event_timestamp" not in working.columns:
            working["event_timestamp"] = pd.Timestamp.utcnow()
        return working

    def _resolve_affinity(self, affinity_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if affinity_df is None:
            if self.affinity_df is None:
                affinity_df = pd.DataFrame({
                    "user_id": self.rng.integers(1, self.config.n_users + 1, size=100),
                    "item_id": self.rng.integers(1, self.config.n_items + 1, size=100),
                    "affinity_score": self.rng.random(100),
                    "candidate_generation_score": self.rng.random(100),
                    "ranking_label": self.rng.random(100),
                })
            else:
                affinity_df = self.affinity_df

        working = affinity_df.copy()
        if "event_timestamp" not in working.columns:
            working["event_timestamp"] = pd.Timestamp.utcnow()
        return working

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        contexts: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate all feature store tables for serving and batch pipelines.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        contexts = self._resolve_contexts(contexts)
        affinity_df = self._resolve_affinity(affinity_df)

        return {
            "user_features": users.copy(),
            "item_features": items.copy(),
            "context_features": contexts.copy(),
            "affinity_features": affinity_df.copy(),
        }


class Phase19FeatureStoreLayer:
    """
    Orchestrator-style wrapper for Phase 19.
    """

    def __init__(
        self,
        foundation,
        users: pd.DataFrame,
        items: pd.DataFrame,
        contexts: pd.DataFrame,
        affinity_df: pd.DataFrame,
    ):
        self.foundation = foundation
        self.users = users
        self.items = items
        self.contexts = contexts
        self.affinity_df = affinity_df
        self.engine = FeatureStoreBuilder(
            foundation,
            users=users,
            items=items,
            contexts=contexts,
            affinity_df=affinity_df,
        )

    def generate(self) -> dict[str, pd.DataFrame]:
        return self.engine.generate(
            users=self.users,
            items=self.items,
            contexts=self.contexts,
            affinity_df=self.affinity_df,
        )


if __name__ == "__main__":
    from foundation import FoundationLayer, GeneratorConfig

    config = GeneratorConfig(
        n_users=10,
        n_items=10,
        embedding_dim=8,
        random_state=42,
    )

    foundation = FoundationLayer(config)

    users = pd.DataFrame({"user_id": [1, 2, 3], "age": [20, 28, 36]})
    items = pd.DataFrame({"item_id": [1, 2], "genre": ["Action", "Drama"]})
    contexts = pd.DataFrame({"context_id": [1, 2], "watch_intent_score": [0.5, 0.7]})
    affinity_df = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [1, 2],
        "affinity_score": [0.9, 0.6],
    })

    builder = FeatureStoreBuilder(
        foundation,
        users=users,
        items=items,
        contexts=contexts,
        affinity_df=affinity_df,
    )

    result = builder.generate(users, items, contexts, affinity_df)
    print(result.keys())
    for name, df in result.items():
        print(name)
        print(df.head())
