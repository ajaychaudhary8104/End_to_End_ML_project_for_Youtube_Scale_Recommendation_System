from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class ColdStartEngine:
    """
    Phase 18 — Cold Start Engine

    Simulates cold-start scenarios for users, items, creators, and genres.

    Output columns include:
    - cold_start_type
    - cold_start_severity
    """

    def __init__(
        self,
        foundation,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        creators: Optional[pd.DataFrame] = None,
        genres: Optional[pd.DataFrame] = None,
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.users = users
        self.items = items
        self.creators = creators
        self.genres = genres

    def _resolve_users(self, users: Optional[pd.DataFrame]) -> pd.DataFrame:
        if users is None:
            if self.users is None:
                users = pd.DataFrame({
                    "user_id": np.arange(1, self.config.n_users + 1),
                    "tenure_days": self.rng.integers(1, 120, size=self.config.n_users),
                    "cold_start_flag": True,
                })
            else:
                users = self.users

        working = users.copy()
        if "user_id" not in working.columns:
            working["user_id"] = np.arange(1, len(working) + 1)
        if "tenure_days" not in working.columns:
            working["tenure_days"] = self.rng.integers(1, 120, size=len(working))
        if "cold_start_flag" not in working.columns:
            working["cold_start_flag"] = working["tenure_days"] <= 30

        return working

    def _resolve_items(self, items: Optional[pd.DataFrame]) -> pd.DataFrame:
        if items is None:
            if self.items is None:
                items = pd.DataFrame({
                    "item_id": np.arange(1, self.config.n_items + 1),
                    "content_age_days": self.rng.integers(1, 30, size=self.config.n_items),
                })
            else:
                items = self.items

        working = items.copy()
        if "item_id" not in working.columns:
            working["item_id"] = np.arange(1, len(working) + 1)
        if "content_age_days" not in working.columns:
            working["content_age_days"] = self.rng.integers(1, 30, size=len(working))

        return working

    def _resolve_creators(self, creators: Optional[pd.DataFrame]) -> pd.DataFrame:
        if creators is None:
            if self.creators is None:
                creators = pd.DataFrame({
                    "creator_id": np.arange(1, max(5, self.config.n_users // 10) + 1),
                    "creator_age_days": self.rng.integers(1, 21, size=max(5, self.config.n_users // 10)),
                })
            else:
                creators = self.creators

        working = creators.copy()
        if "creator_id" not in working.columns:
            working["creator_id"] = np.arange(1, len(working) + 1)
        if "creator_age_days" not in working.columns:
            working["creator_age_days"] = self.rng.integers(1, 21, size=len(working))

        return working

    def _resolve_genres(self, genres: Optional[pd.DataFrame]) -> pd.DataFrame:
        if genres is None:
            if self.genres is None:
                genres = pd.DataFrame({
                    "genre": ["Action", "Drama", "Comedy", "SciFi", "Sports"],
                    "genre_age_days": self.rng.integers(1, 21, size=5),
                })
            else:
                genres = self.genres

        working = genres.copy()
        if "genre" not in working.columns:
            working["genre"] = [f"genre_{i}" for i in range(1, len(working) + 1)]
        if "genre_age_days" not in working.columns:
            working["genre_age_days"] = self.rng.integers(1, 21, size=len(working))

        return working

    def _severity_from_age(self, age_days: int, max_age: int = 30) -> float:
        return float(np.clip(1.0 - (age_days / max_age), 0.0, 1.0))

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        creators: Optional[pd.DataFrame] = None,
        genres: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate a cold-start simulation dataset.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        creators = self._resolve_creators(creators)
        genres = self._resolve_genres(genres)

        rows = []

        for _, user in users.iterrows():
            if bool(user["cold_start_flag"]):
                rows.append({
                    "entity_type": "user",
                    "entity_id": int(user["user_id"]),
                    "cold_start_type": "new_user",
                    "cold_start_severity": self._severity_from_age(
                        int(user["tenure_days"]),
                        max_age=30
                    ),
                })

        for _, item in items.iterrows():
            if int(item["content_age_days"]) <= 14:
                rows.append({
                    "entity_type": "item",
                    "entity_id": int(item["item_id"]),
                    "cold_start_type": "new_item",
                    "cold_start_severity": self._severity_from_age(
                        int(item["content_age_days"]),
                        max_age=14
                    ),
                })

        for _, creator in creators.iterrows():
            if int(creator["creator_age_days"]) <= 14:
                rows.append({
                    "entity_type": "creator",
                    "entity_id": int(creator["creator_id"]),
                    "cold_start_type": "new_creator",
                    "cold_start_severity": self._severity_from_age(
                        int(creator["creator_age_days"]),
                        max_age=14
                    ),
                })

        for _, genre in genres.iterrows():
            if int(genre["genre_age_days"]) <= 14:
                rows.append({
                    "entity_type": "genre",
                    "entity_id": int(genre.name) + 1,
                    "cold_start_type": "new_genre",
                    "cold_start_severity": self._severity_from_age(
                        int(genre["genre_age_days"]),
                        max_age=14
                    ),
                })

        return pd.DataFrame(rows)


class Phase18ColdStartLayer:
    """
    Orchestrator-style wrapper for Phase 18.
    """

    def __init__(
        self,
        foundation,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        creators: Optional[pd.DataFrame] = None,
        genres: Optional[pd.DataFrame] = None,
    ):
        self.foundation = foundation
        self.users = users
        self.items = items
        self.creators = creators
        self.genres = genres
        self.engine = ColdStartEngine(
            foundation,
            users=users,
            items=items,
            creators=creators,
            genres=genres,
        )

    def generate(self) -> pd.DataFrame:
        return self.engine.generate(
            users=self.users,
            items=self.items,
            creators=self.creators,
            genres=self.genres,
        )


if __name__ == "__main__":
    from foundation import FoundationLayer, GeneratorConfig

    config = GeneratorConfig(
        n_users=10,
        n_items=20,
        embedding_dim=8,
        random_state=42,
    )

    foundation = FoundationLayer(config)

    users = pd.DataFrame({
        "user_id": [1, 2, 3],
        "tenure_days": [5, 12, 80],
        "cold_start_flag": [True, True, False],
    })
    items = pd.DataFrame({
        "item_id": [1, 2, 3],
        "content_age_days": [3, 10, 40],
    })
    creators = pd.DataFrame({
        "creator_id": [1, 2],
        "creator_age_days": [4, 20],
    })
    genres = pd.DataFrame({
        "genre": ["Action", "Drama"],
        "genre_age_days": [7, 25],
    })

    generator = ColdStartEngine(foundation, users, items, creators, genres)
    result = generator.generate(users, items, creators, genres)
    print(result.head())
