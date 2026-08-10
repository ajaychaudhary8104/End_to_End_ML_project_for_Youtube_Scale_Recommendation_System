from __future__ import annotations

import numpy as np
import pandas as pd

from ...foundation import FoundationLayer


class SessionTimelineGenerator:
	EVENT_TYPES = ["browse", "search", "click", "watch", "skip", "review", "share"]

	def __init__(self, foundation: FoundationLayer, users=None, contexts=None):
		self.foundation = foundation
		self.config = foundation.config
		self.rng = foundation.rng
		self.users = users
		self.contexts = contexts

	def generate(self, users=None, contexts=None) -> pd.DataFrame:
		users = users if users is not None else self.users
		if users is None or users.empty:
			raise ValueError("users must be provided")
		contexts = contexts if contexts is not None else self.contexts
		if contexts is None or contexts.empty:
			timestamps = pd.Series(self.rng.choice(pd.date_range(self.config.start_date, self.config.end_date, freq="h"), len(users)))
		else:
			timestamps = contexts["timestamp"] if "timestamp" in contexts else pd.Series(self.rng.choice(pd.date_range(self.config.start_date, self.config.end_date, freq="h"), len(users)))
		rows = []
		counter = 1
		for user_index, user in users.reset_index(drop=True).iterrows():
			count = max(1, int(user.get("session_frequency", self.rng.integers(1, 5))))
			for _ in range(count):
				events = [self.rng.choice(["browse", "search", "click"])] + list(self.rng.choice(self.EVENT_TYPES, self.rng.integers(1, 6)))
				length = max(30, int(len(events) * 35 + self.rng.integers(0, 180)))
				start = timestamps.iloc[int(self.rng.integers(0, len(timestamps)))]
				rows.append({"session_id": f"S{counter:09d}", "user_id": int(user["user_id"]), "session_start": start, "session_end": start + pd.to_timedelta(length, unit="s"), "session_length": length, "session_events": events, "event_count": len(events), "first_event": events[0], "last_event": events[-1]})
				counter += 1
		return pd.DataFrame(rows)


class Phase8SessionTimelineLayer:
	def __init__(self, foundation, users, contexts=None):
		self.engine = SessionTimelineGenerator(foundation, users, contexts)
		self.users, self.contexts = users, contexts

	def generate(self):
		return self.engine.generate(self.users, self.contexts)
