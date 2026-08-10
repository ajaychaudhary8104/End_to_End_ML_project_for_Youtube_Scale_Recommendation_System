from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class SequentialRecommendationGenerator:
    """
    Phase 20 — Sequential Recommendation Datasets

    Creates:
    - SASRec-style sequence datasets with user_sequence and next_item_label
    - BERT4Rec-style masked datasets with masked_sequence and target_item
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
                    "genre": self.rng.choice([
                        "Action", "Drama", "Comedy", "SciFi", "Sports"
                    ], size=self.config.n_items),
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
                    "user_id": self.rng.integers(1, self.config.n_users + 1, size=100),
                    "item_id": self.rng.integers(1, self.config.n_items + 1, size=100),
                })
            else:
                interactions = self.interactions

        return interactions.copy()

    def build_sasrec_dataset(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        interactions: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Create a SASRec-style dataset using per-user ordered item sequences.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        interactions = self._resolve_interactions(interactions)

        rows = []
        for _, user in users.iterrows():
            sequence = items.sample(
                n=min(5, len(items)),
                replace=False,
                random_state=int(self.rng.integers(0, 1_000_000))
            )["item_id"].tolist()
            next_item = sequence[-1]
            rows.append({
                "user_id": int(user["user_id"]),
                "user_sequence": sequence,
                "next_item_label": int(next_item),
            })

        return pd.DataFrame(rows)

    def build_bert4rec_dataset(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        interactions: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Create a BERT4Rec-style masked sequence dataset.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        interactions = self._resolve_interactions(interactions)

        rows = []
        for _, user in users.iterrows():
            sequence = items.sample(
                n=min(5, len(items)),
                replace=False,
                random_state=int(self.rng.integers(0, 1_000_000))
            )["item_id"].tolist()
            masked_index = self.rng.integers(0, len(sequence))
            masked_sequence = sequence.copy()
            target_item = masked_sequence[masked_index]
            masked_sequence[masked_index] = -1

            rows.append({
                "user_id": int(user["user_id"]),
                "masked_sequence": masked_sequence,
                "target_item": int(target_item),
            })

        return pd.DataFrame(rows)

    def generate(self) -> dict[str, pd.DataFrame]:
        """
        Return both sequential recommendation datasets.
        """

        return {
            "sasrec_dataset": self.build_sasrec_dataset(),
            "bert4rec_dataset": self.build_bert4rec_dataset(),
        }


class Phase20SequentialRecommendationLayer:
    """
    Orchestrator-style wrapper for Phase 20.
    """

    def __init__(
        self,
        foundation,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: Optional[pd.DataFrame] = None,
    ):
        self.foundation = foundation
        self.users = users
        self.items = items
        self.interactions = interactions
        self.engine = SequentialRecommendationGenerator(
            foundation,
            users=users,
            items=items,
            interactions=interactions,
        )

    def generate(self) -> dict[str, pd.DataFrame]:
        return self.engine.generate()


if __name__ == "__main__":
    from foundation import FoundationLayer, GeneratorConfig

    config = GeneratorConfig(
        n_users=3,
        n_items=10,
        embedding_dim=8,
        random_state=42,
    )

    foundation = FoundationLayer(config)
    users = pd.DataFrame({"user_id": [1, 2, 3]})
    items = pd.DataFrame({"item_id": [1, 2, 3, 4, 5, 6]})

    generator = SequentialRecommendationGenerator(
        foundation,
        users=users,
        items=items,
    )

    result = generator.generate()
    print(result["sasrec_dataset"].head())
    print(result["bert4rec_dataset"].head())
