from __future__ import annotations

import numpy as np
import pandas as pd


class SearchGenerator:
	def __init__(self, foundation, users=None, items=None, affinity_df=None):
		self.foundation, self.config, self.rng = foundation, foundation.config, foundation.rng
		self.users, self.items, self.affinity_df = users, items, affinity_df

	def generate(self, users=None, items=None, affinity_df=None):
		users = users if users is not None else self.users
		items = items if items is not None else self.items
		affinity_df = affinity_df if affinity_df is not None else self.affinity_df
		if users is None or items is None or affinity_df is None:
			raise ValueError("users, items, and affinity_df must be provided")
		rows = []
		for user in users.itertuples(index=False):
			for item in items.head(3).itertuples(index=False):
				match = affinity_df[(affinity_df["user_id"] == user.user_id) & (affinity_df["item_id"] == item.item_id)]
				score_column = "search_click_score" if "search_click_score" in affinity_df else "click_probability"
				score = float(match[score_column].mean()) if not match.empty and score_column in match else float(self.rng.random())
				rows.append({"query": f"query_user_{int(user.user_id)}_item_{int(item.item_id)}", "query_embedding": self.rng.normal(size=self.config.embedding_dim).astype(np.float32), "clicked_item": int(item.item_id), "user_id": int(user.user_id), "search_click_score": np.clip(score, 0, 1)})
		return pd.DataFrame(rows)


class Phase16SearchLayer:
	def __init__(self, foundation, users, items, affinity_df):
		self.engine = SearchGenerator(foundation, users, items, affinity_df)

	def generate(self):
		return self.engine.generate()
