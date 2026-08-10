from __future__ import annotations

import numpy as np
import pandas as pd


class SequentialRecommendationGenerator:
	def __init__(self, foundation, users=None, items=None, interactions=None):
		self.foundation, self.rng = foundation, foundation.rng
		self.users, self.items, self.interactions = users, items, interactions

	def _sequence(self, user_id, items):
		if self.interactions is not None and not self.interactions.empty and "user_id" in self.interactions:
			sequence = self.interactions[self.interactions["user_id"] == user_id].get("item_id", pd.Series()).tolist()
		else:
			sequence = []
		if len(sequence) < 2:
			sequence = items.sample(min(5, len(items)), random_state=int(self.rng.integers(0, 1_000_000)))["item_id"].tolist()
		return sequence

	def build_sasrec_dataset(self, users=None, items=None, interactions=None):
		users = users if users is not None else self.users
		items = items if items is not None else self.items
		if users is None or items is None:
			raise ValueError("users and items must be provided")
		previous = self.interactions
		if interactions is not None:
			self.interactions = interactions
		rows = []
		for user in users.itertuples(index=False):
			sequence = self._sequence(user.user_id, items)
			rows.append({"user_id": int(user.user_id), "user_sequence": sequence[:-1], "next_item_label": int(sequence[-1])})
		self.interactions = previous
		return pd.DataFrame(rows)

	def build_bert4rec_dataset(self, users=None, items=None, interactions=None):
		result = self.build_sasrec_dataset(users, items, interactions)
		result = result.rename(columns={"user_sequence": "masked_sequence", "next_item_label": "target_item"})
		result["masked_sequence"] = result["masked_sequence"].apply(lambda sequence: sequence[:-1] + [-1] if sequence else [-1])
		return result

	def generate(self):
		return {"sasrec_dataset": self.build_sasrec_dataset(), "bert4rec_dataset": self.build_bert4rec_dataset()}


class Phase20SequentialRecommendationLayer:
	def __init__(self, foundation, users, items, interactions=None):
		self.engine = SequentialRecommendationGenerator(foundation, users, items, interactions)

	def generate(self):
		return self.engine.generate()
