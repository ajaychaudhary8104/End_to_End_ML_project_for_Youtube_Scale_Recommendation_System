from __future__ import annotations

import numpy as np
import pandas as pd

from ..foundation import FoundationLayer


class RetrievalGenerator:
	def __init__(self, foundation: FoundationLayer, users=None, items=None, affinity_df=None):
		self.foundation, self.config, self.rng = foundation, foundation.config, foundation.rng
		self.users, self.items, self.affinity_df = users, items, affinity_df

	def generate_two_tower_dataset(self, users, items, affinity_df):
		rows = []
		for _, user in users.iterrows():
			for _, item in items.head(5).iterrows():
				match = affinity_df[(affinity_df["user_id"] == user["user_id"]) & (affinity_df["item_id"] == item["item_id"])]
				label = float(match["candidate_generation_score"].mean()) if not match.empty and "candidate_generation_score" in match else float(self.rng.random())
				rows.append({"user_id": int(user["user_id"]), "item_id": int(item["item_id"]), "user_features": f"u{int(user['user_id'])}", "item_features": f"i{int(item['item_id'])}", "retrieval_label": np.clip(label, 0, 1)})
		return pd.DataFrame(rows)

	def generate_ann_dataset(self, users, items, affinity_df):
		rows = []
		for _, user in users.iterrows():
			for _, item in items.head(3).iterrows():
				rows.append({"user_id": int(user["user_id"]), "item_id": int(item["item_id"]), "query_embedding": self.rng.normal(size=self.config.embedding_dim).astype(np.float32), "candidate_embedding": self.rng.normal(size=self.config.embedding_dim).astype(np.float32), "retrieval_label": float(self.rng.random())})
		return pd.DataFrame(rows)

	def generate(self, users=None, items=None, affinity_df=None):
		users = users if users is not None else self.users
		items = items if items is not None else self.items
		affinity_df = affinity_df if affinity_df is not None else self.affinity_df
		if users is None or items is None or affinity_df is None:
			raise ValueError("users, items, and affinity_df must be provided")
		return {"two_tower_dataset": self.generate_two_tower_dataset(users, items, affinity_df), "candidate_dataset": self.generate_ann_dataset(users, items, affinity_df)}


class Phase13RetrievalLayer:
	def __init__(self, foundation, users, items, affinity_df):
		self.engine = RetrievalGenerator(foundation, users, items, affinity_df)

	def generate(self):
		return self.engine.generate()
