from __future__ import annotations

import numpy as np
import pandas as pd

from ..foundation import FoundationLayer


class ColdStartEngine:
    def __init__(self, foundation: FoundationLayer, users=None, items=None, creators=None, genres=None):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.users, self.items, self.creators, self.genres = users, items, creators, genres

    def generate(self, users=None, items=None, creators=None, genres=None) -> pd.DataFrame:
        users = users if users is not None else self.users
        items = items if items is not None else self.items
        creators = creators if creators is not None else self.creators
        genres = genres if genres is not None else self.genres
        if users is None:
            users = pd.DataFrame({"user_id": np.arange(1, self.config.n_users + 1), "tenure_days": self.rng.integers(1, 120, self.config.n_users)})
        if items is None:
            items = pd.DataFrame({"item_id": np.arange(1, self.config.n_items + 1), "content_age_days": self.rng.integers(1, 30, self.config.n_items)})
        users = users.copy()
        items = items.copy()
        if "tenure_days" not in users:
            users["tenure_days"] = self.rng.integers(1, 120, len(users))
        if "content_age_days" not in items:
            items["content_age_days"] = self.rng.integers(1, 30, len(items))
        rows = []
        for row in users[users["tenure_days"] <= 30].itertuples(index=False):
            rows.append({"entity_type": "user", "entity_id": int(row.user_id), "cold_start_type": "new_user", "cold_start_severity": float(np.clip(1 - row.tenure_days / 30, 0, 1))})
        for row in items[items["content_age_days"] <= 14].itertuples(index=False):
            rows.append({"entity_type": "item", "entity_id": int(row.item_id), "cold_start_type": "new_item", "cold_start_severity": float(np.clip(1 - row.content_age_days / 14, 0, 1))})
        if creators is not None:
            creators = creators.copy()
            if "creator_age_days" in creators:
                for row in creators[creators["creator_age_days"] <= 14].itertuples(index=False):
                    rows.append({"entity_type": "creator", "entity_id": int(row.creator_id), "cold_start_type": "new_creator", "cold_start_severity": float(np.clip(1 - row.creator_age_days / 14, 0, 1))})
        if genres is not None and "genre_age_days" in genres:
            for index, row in genres[genres["genre_age_days"] <= 14].iterrows():
                rows.append({"entity_type": "genre", "entity_id": int(index) + 1, "cold_start_type": "new_genre", "cold_start_severity": float(np.clip(1 - row["genre_age_days"] / 14, 0, 1))})
        return pd.DataFrame(
            rows,
            columns=["entity_type", "entity_id", "cold_start_type", "cold_start_severity"],
        )


class Phase18ColdStartLayer:
    def __init__(self, foundation, users=None, items=None, creators=None, genres=None):
        self.engine = ColdStartEngine(foundation, users, items, creators, genres)

    def generate(self):
        return self.engine.generate()
