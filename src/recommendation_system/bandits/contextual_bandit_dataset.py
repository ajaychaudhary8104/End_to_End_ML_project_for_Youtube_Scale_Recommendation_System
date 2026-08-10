from __future__ import annotations

import numpy as np
import pandas as pd


class BanditGenerator:
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
			for item in items.head(3).itertuples(index=False):
				match = affinity_df[(affinity_df["user_id"] == user.user_id) & (affinity_df["item_id"] == item.item_id)]
				reward = float(match["bandit_reward"].mean()) if not match.empty and "bandit_reward" in match else float(self.rng.random())
				rows.append({"user_id": int(user.user_id), "action": int(item.item_id), "reward": np.clip(reward, 0, 1), "policy_probability": float(np.clip(self.rng.random(), 0.05, 0.95))})
		return pd.DataFrame(rows)


class Phase17BanditLayer:
	def __init__(self, foundation, users, items, affinity_df):
		self.engine = BanditGenerator(foundation, users, items, affinity_df)

	def generate(self):
		return self.engine.generate()
