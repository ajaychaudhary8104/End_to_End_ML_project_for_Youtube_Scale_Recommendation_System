from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class WatchHistoryGenerator:
    """
    Phase 10 — Watch History Generator

    Builds historical user consumption records with watch-level
    behavior statistics.

    Outputs:
    - watch_timestamp
    - watch_duration
    - completion_rate
    - rewatch_count
    """

    def __init__(
        self,
        foundation,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        interactions: Optional[pd.DataFrame] = None
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.users = users
        self.items = items
        self.interactions = interactions

    def _resolve_users(
        self,
        users: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if users is None:
            if self.users is None:
                raise ValueError("users must be provided")
            users = self.users

        working = users.copy()
        if "user_id" not in working.columns:
            working["user_id"] = np.arange(1, len(working) + 1)

        return working

    def _resolve_items(
        self,
        items: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if items is None:
            if self.items is None:
                items = pd.DataFrame(
                    {
                        "item_id": np.arange(1, self.config.n_items + 1)
                    }
                )
            else:
                items = self.items

        working = items.copy()
        if "item_id" not in working.columns:
            working["item_id"] = np.arange(1, len(working) + 1)

        return working

    def _resolve_interactions(
        self,
        interactions: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if interactions is None:
            if self.interactions is None:
                interactions = pd.DataFrame(
                    {
                        "user_id": self.rng.integers(
                            1,
                            self.config.n_users + 1,
                            size=100
                        ),
                        "item_id": self.rng.integers(
                            1,
                            self.config.n_items + 1,
                            size=100
                        )
                    }
                )
            else:
                interactions = self.interactions

        working = interactions.copy()
        if "watch" not in working.columns:
            working["watch"] = self.rng.integers(
                0,
                2,
                size=len(working)
            )

        return working

    def _generate_watch_timestamp(
        self,
        n_rows: int
    ) -> pd.Series:
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        total_seconds = int((end - start).total_seconds())

        offsets = self.rng.integers(
            0,
            total_seconds,
            size=n_rows
        )

        return pd.Series(
            start + pd.to_timedelta(offsets, unit="s")
        )

    def _generate_watch_duration(
        self,
        n_rows: int
    ) -> np.ndarray:
        sample = self.rng.lognormal(
            mean=3.6,
            sigma=0.8,
            size=n_rows
        )

        return np.clip(
            np.round(sample),
            1,
            3600
        ).astype(np.int32)

    def _generate_completion_rate(
        self,
        n_rows: int
    ) -> np.ndarray:
        rates = self.rng.beta(
            2.5,
            1.5,
            size=n_rows
        )

        return np.clip(rates, 0.0, 1.0)

    def _generate_rewatch_count(
        self,
        n_rows: int
    ) -> np.ndarray:
        counts = self.rng.poisson(
            lam=0.6,
            size=n_rows
        )

        return np.clip(counts, 0, 20).astype(np.int32)

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        interactions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate historical watch records.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        interactions = self._resolve_interactions(interactions)

        n_rows = len(interactions)

        history = pd.DataFrame(
            {
                "watch_timestamp": self._generate_watch_timestamp(n_rows),
                "watch_duration": self._generate_watch_duration(n_rows),
                "completion_rate": self._generate_completion_rate(n_rows),
                "rewatch_count": self._generate_rewatch_count(n_rows),
            }
        )

        history["user_id"] = (
            interactions["user_id"].to_numpy()
        )
        history["item_id"] = (
            interactions["item_id"].to_numpy()
        )

        if "watch" in interactions.columns:
            history["watch_flag"] = interactions["watch"].to_numpy()

        history["watch_duration"] = history["watch_duration"].astype(np.int32)
        history["rewatch_count"] = history["rewatch_count"].astype(np.int32)

        return history


class Phase10WatchHistoryLayer:
    """
    Orchestrator-style wrapper for Phase 10.
    """

    def __init__(
        self,
        foundation,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: pd.DataFrame
    ):
        self.foundation = foundation
        self.users = users
        self.items = items
        self.interactions = interactions
        self.engine = WatchHistoryGenerator(
            foundation,
            users=users,
            items=items,
            interactions=interactions
        )

    def generate(self) -> pd.DataFrame:
        return self.engine.generate(
            users=self.users,
            items=self.items,
            interactions=self.interactions
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
    items = pd.DataFrame({"item_id": [1, 2, 3]})
    interactions = pd.DataFrame(
        {
            "user_id": [1, 2, 2, 3],
            "item_id": [1, 2, 3, 1],
            "watch": [1, 1, 0, 1]
        }
    )

    generator = WatchHistoryGenerator(
        foundation,
        users=users,
        items=items,
        interactions=interactions
    )

    result = generator.generate(users, items, interactions)
    print(result.head())
