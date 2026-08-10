from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class EventStreamGenerator:
    """
    Phase 11 — Event Stream Generator

    Produces streaming event logs compatible with modern event systems.
    This lightweight generator emits recommendation impressions and user
    interaction events in a streaming-friendly format.

    Examples:
    - view
    - click
    - scroll
    - hover
    - search
    - purchase
    """

    EVENT_TYPES = [
        "view",
        "click",
        "scroll",
        "hover",
        "search",
        "purchase"
    ]

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
        if "event_type" not in working.columns:
            working["event_type"] = self.rng.choice(
                self.EVENT_TYPES,
                size=len(working)
            )

        return working

    def _generate_event_timestamp(
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

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        interactions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate an event stream dataset.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        interactions = self._resolve_interactions(interactions)

        stream = pd.DataFrame(
            {
                "event_timestamp": self._generate_event_timestamp(len(interactions)),
                "event_type": interactions["event_type"].to_numpy(),
                "user_id": interactions["user_id"].to_numpy(),
                "item_id": interactions["item_id"].to_numpy(),
                "event_source": self.rng.choice(
                    ["web", "mobile", "tv", "api"],
                    size=len(interactions)
                ),
                "session_id": [
                    f"S{i:09d}" for i in range(1, len(interactions) + 1)
                ]
            }
        )

        return stream


class Phase11EventStreamLayer:
    """
    Orchestrator-style wrapper for Phase 11.
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
        self.engine = EventStreamGenerator(
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
            "event_type": ["view", "click", "scroll", "purchase"]
        }
    )

    generator = EventStreamGenerator(
        foundation,
        users=users,
        items=items,
        interactions=interactions
    )

    result = generator.generate(users, items, interactions)
    print(result.head())
