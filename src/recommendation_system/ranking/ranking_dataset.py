from __future__ import annotations

import numpy as np
import pandas as pd


class RankingGenerator:
	def __init__(self, foundation, users=None, items=None, affinity_df=None):
		self.foundation, self.rng = foundation, foundation.rng
		self.users, self.items, self.affinity_df = users, items, affinity_df

	def generate(self, users=None, items=None, affinity_df=None):
		users = users if users is not None else self.users
		items = items if items is not None else self.items
		affinity_df = affinity_df if affinity_df is not None else self.affinity_df
		if users is None or items is None or affinity_df is None:
			raise ValueError("users, items, and affinity_df must be provided")
		rows = []
		for user in users.itertuples(index=False):
			subset = items.sample(min(5, len(items)), random_state=int(self.rng.integers(0, 1_000_000)))
			for item in subset.itertuples(index=False):
				match = affinity_df[(affinity_df["user_id"] == user.user_id) & (affinity_df["item_id"] == item.item_id)]
				label = float(match["ranking_label"].mean()) if not match.empty and "ranking_label" in match else float(self.rng.random())
				rows.append({"query_id": f"q{int(user.user_id):09d}", "user_id": int(user.user_id), "item_id": int(item.item_id), "rank_label": np.clip(label, 0, 1)})
		return pd.DataFrame(rows)


class Phase14RankingLayer:
	def __init__(self, foundation, users, items, affinity_df):
		self.engine = RankingGenerator(foundation, users, items, affinity_df)

	def generate(self):
		return self.engine.generate()
