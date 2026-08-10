from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class BanditGenerator:
    """
    Phase 17 — Bandit Generator

    Creates contextual bandit training data.

    Output schema includes:
    - action
    - reward
    - policy_probability
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
                    "bandit_reward": self.rng.random(100),
                    "satisfaction_probability": self.rng.random(100),
                    "retention_probability": self.rng.random(100),
                })
            else:
                affinity_df = self.affinity_df

        return affinity_df.copy()

    def _policy_probability(self, user_id: int, item_id: int) -> float:
        """
        Synthetic policy probability for a given action.
        """

        return float(np.clip(self.rng.random(), 0.05, 0.95))

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate a contextual bandit training dataset.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        affinity_df = self._resolve_affinity(affinity_df)

        rows = []
        for _, user in users.iterrows():
            for item in items.head(3).itertuples(index=False):
                item_id = int(item.item_id)

                reward = float(
                    affinity_df.loc[
                        (affinity_df["user_id"] == int(user["user_id"]))
                        & (affinity_df["item_id"] == item_id),
                        "bandit_reward"
                    ].mean()
                    if (
                        "bandit_reward" in affinity_df.columns
                        and not affinity_df.loc[
                            (affinity_df["user_id"] == int(user["user_id"]))
                            & (affinity_df["item_id"] == item_id)
                        ].empty
                    )
                    else self.rng.random()
                )

                rows.append({
                    "user_id": int(user["user_id"]),
                    "action": item_id,
                    "reward": float(np.clip(reward, 0.0, 1.0)),
                    "policy_probability": self._policy_probability(
                        int(user["user_id"]),
                        item_id
                    ),
                })

        return pd.DataFrame(rows)


class Phase17BanditLayer:
    """
    Orchestrator-style wrapper for Phase 17.
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
        self.engine = BanditGenerator(
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
        "bandit_reward": [0.9, 0.6, 0.7, 0.8, 0.5]
    })

    generator = BanditGenerator(
        foundation,
        users=users,
        items=items,
        affinity_df=affinity_df
    )

    result = generator.generate(users, items, affinity_df)
    print(result.head())
