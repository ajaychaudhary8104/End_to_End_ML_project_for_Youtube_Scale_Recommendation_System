from __future__ import annotations

import pandas as pd


class FeatureStoreBuilder:
    def __init__(self, foundation, users=None, items=None, contexts=None, affinity_df=None):
        self.foundation = foundation
        self.users, self.items, self.contexts, self.affinity_df = users, items, contexts, affinity_df

    @staticmethod
    def _with_timestamp(frame: pd.DataFrame, timestamp_column: str = "event_timestamp") -> pd.DataFrame:
        result = frame.copy()
        if timestamp_column not in result:
            result[timestamp_column] = pd.Timestamp.utcnow()
        return result

    def generate(self, users=None, items=None, contexts=None, affinity_df=None) -> dict[str, pd.DataFrame]:
        users = users if users is not None else self.users
        items = items if items is not None else self.items
        contexts = contexts if contexts is not None else self.contexts
        affinity_df = affinity_df if affinity_df is not None else self.affinity_df
        if any(frame is None or frame.empty for frame in [users, items, contexts, affinity_df]):
            raise ValueError("users, items, contexts, and affinity_df must be non-empty")
        return {
            "user_features": self._with_timestamp(users),
            "item_features": self._with_timestamp(items),
            "context_features": self._with_timestamp(contexts),
            "affinity_features": self._with_timestamp(affinity_df),
        }


class Phase25FeatureStoreLayer:
    def __init__(self, foundation, users, items, contexts, affinity_df):
        self.engine = FeatureStoreBuilder(foundation, users, items, contexts, affinity_df)

    def generate(self):
        return self.engine.generate()
