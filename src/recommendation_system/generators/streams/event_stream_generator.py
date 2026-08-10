from __future__ import annotations

import numpy as np
import pandas as pd

from ...foundation import FoundationLayer


class EventStreamGenerator:
	EVENT_TYPES = ["view", "click", "scroll", "hover", "search", "purchase"]

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
		timestamps = start + pd.to_timedelta(self.rng.integers(0, seconds, len(interactions)), unit="s")
		return pd.DataFrame({"event_timestamp": timestamps, "event_type": self.rng.choice(self.EVENT_TYPES, len(interactions)), "user_id": interactions["user_id"].to_numpy(), "item_id": interactions["item_id"].to_numpy(), "event_source": self.rng.choice(["web", "mobile", "tv", "api"], len(interactions)), "session_id": interactions.get("session_id", pd.Series([f"S{i:09d}" for i in range(1, len(interactions) + 1)])).to_numpy()})


class Phase11EventStreamLayer:
	def __init__(self, foundation, users, items, interactions):
		self.engine = EventStreamGenerator(foundation, users, items, interactions)

	def generate(self):
		return self.engine.generate()
