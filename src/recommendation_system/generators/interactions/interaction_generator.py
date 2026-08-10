from __future__ import annotations

import numpy as np
import pandas as pd

from ...foundation import FoundationLayer


class InteractionGenerator:
	EVENT_COLUMNS = ["impression", "click", "watch", "completion", "like", "dislike", "share", "save"]

	def __init__(self, foundation: FoundationLayer, sessions=None, affinity_df=None, items=None):
		self.foundation = foundation
		self.config = foundation.config
		self.rng = foundation.rng
		self.sessions, self.affinity_df, self.items = sessions, affinity_df, items

	def generate(self, sessions=None, affinity_df=None, items=None) -> pd.DataFrame:
		sessions = sessions if sessions is not None else self.sessions
		if sessions is None or sessions.empty:
			raise ValueError("sessions must be provided")
		items = items if items is not None else self.items
		if items is None or items.empty:
			items = pd.DataFrame({"item_id": np.arange(1, self.config.n_items + 1)})
		affinity_df = affinity_df if affinity_df is not None else self.affinity_df
		rows = []
		for _, session in sessions.iterrows():
			candidates = affinity_df[affinity_df["user_id"] == session["user_id"]] if affinity_df is not None and "user_id" in affinity_df else pd.DataFrame()
			if not candidates.empty:
				item_id = candidates.iloc[int(self.rng.integers(0, len(candidates)))] ["item_id"]
				score = float(candidates.iloc[0].get("click_probability", 0.4))
			else:
				item = items.iloc[int(self.rng.integers(0, len(items)))]
				item_id, score = item["item_id"], 0.4 + float(item.get("popularity_score", 0.5)) * 0.1
			draw = self.rng.random()
			click = int(draw < score)
			watch = int(draw < max(0.15, score * 0.70))
			completion = int(draw < max(0.10, score * 0.40))
			like = int(draw < max(0.05, completion * 0.35))
			rows.append({"session_id": session["session_id"], "user_id": int(session["user_id"]), "item_id": int(item_id), "session_length": int(session["session_length"]), "impression": 1, "click": click, "watch": watch, "completion": completion, "like": like, "dislike": int(draw < max(0.01, (1 - completion) * 0.12)), "share": int(draw < max(0.01, like * 0.25)), "save": int(draw < max(0.02, like * 0.40))})
		return pd.DataFrame(rows).astype({column: "int8" for column in self.EVENT_COLUMNS})


class Phase9InteractionLayer:
	def __init__(self, foundation, sessions, affinity_df=None, items=None):
		self.engine = InteractionGenerator(foundation, sessions, affinity_df, items)

	def generate(self):
		return self.engine.generate()
