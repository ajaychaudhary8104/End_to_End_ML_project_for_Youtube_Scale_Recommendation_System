from __future__ import annotations

import numpy as np
import pandas as pd

from ..foundation import FoundationLayer


class WatchHistoryGenerator:
    def __init__(self, foundation: FoundationLayer, users=None, items=None, interactions=None):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.users, self.items, self.interactions = users, items, interactions

    def generate(self, users=None, items=None, interactions=None) -> pd.DataFrame:
        interactions = interactions if interactions is not None else self.interactions
        if interactions is None or interactions.empty:
            raise ValueError("interactions must be provided")
        start, end = pd.Timestamp(self.config.start_date), pd.Timestamp(self.config.end_date)
        seconds = max(int((end - start).total_seconds()), 1)
        rows = len(interactions)
        return pd.DataFrame({
            "watch_timestamp": start + pd.to_timedelta(self.rng.integers(0, seconds, rows), unit="s"),
            "user_id": interactions["user_id"].to_numpy(),
            "item_id": interactions["item_id"].to_numpy(),
            "watch_duration": np.clip(np.round(self.rng.lognormal(3.6, 0.8, rows)), 1, 3600).astype(np.int32),
            "completion_rate": np.clip(self.rng.beta(2.5, 1.5, rows), 0, 1),
            "rewatch_count": np.clip(self.rng.poisson(0.6, rows), 0, 20).astype(np.int32),
            "watch_flag": interactions.get("watch", pd.Series(np.ones(rows, dtype=np.int8))).to_numpy(),
        })


class Phase10WatchHistoryLayer:
    def __init__(self, foundation, users, items, interactions):
        self.engine = WatchHistoryGenerator(foundation, users, items, interactions)

    def generate(self):
        return self.engine.generate()
