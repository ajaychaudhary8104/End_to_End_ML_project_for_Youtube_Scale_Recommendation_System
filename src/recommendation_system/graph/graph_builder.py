from __future__ import annotations

import numpy as np
import pandas as pd


class GraphRecommendationBuilder:
	def __init__(self, foundation, users=None, items=None, interactions=None):
		self.foundation, self.rng = foundation, foundation.rng
		self.users, self.items, self.interactions = users, items, interactions

	def _resolve(self, users, items, interactions):
		users = users if users is not None else self.users
		items = items if items is not None else self.items
		interactions = interactions if interactions is not None else self.interactions
		if users is None or items is None or interactions is None:
			raise ValueError("users, items, and interactions must be provided")
		interactions = interactions.copy()
		if "interaction_weight" not in interactions:
			interactions["interaction_weight"] = interactions.get("watch", pd.Series(np.ones(len(interactions)))).astype(float)
		return users, items, interactions

	def build_adjacency_matrix(self, n_users, n_items, interactions):
		matrix = np.zeros((n_users, n_items), dtype=np.float32)
		user_ids = {value: index for index, value in enumerate(self.users["user_id"])}
		item_ids = {value: index for index, value in enumerate(self.items["item_id"])}
		for row in interactions.itertuples(index=False):
			if row.user_id in user_ids and row.item_id in item_ids:
				matrix[user_ids[row.user_id], item_ids[row.item_id]] = float(row.interaction_weight)
		return pd.DataFrame(matrix)

	def build_edge_index(self, interactions):
		edges = interactions[["user_id", "item_id", "interaction_weight"]].copy()
		edges["user_node"] = pd.factorize(edges["user_id"])[0]
		edges["item_node"] = edges["item_id"]
		return edges

	def build_graph_features(self, users, items):
		rows = []
		for index, row in users.reset_index(drop=True).iterrows():
			rows.append({"node_id": index, "node_type": "user", "feature_scalar": float(row.get("engagement_score", 0.5))})
		offset = len(users)
		for index, row in items.reset_index(drop=True).iterrows():
			rows.append({"node_id": offset + index, "node_type": "item", "feature_scalar": float(row.get("popularity_score", 0.5))})
		return pd.DataFrame(rows)

	def generate(self, users=None, items=None, interactions=None):
		users, items, interactions = self._resolve(users, items, interactions)
		self.users, self.items = users, items
		return {"adjacency_matrix": self.build_adjacency_matrix(len(users), len(items), interactions), "edge_index": self.build_edge_index(interactions), "graph_features": self.build_graph_features(users, items)}


class Phase21GraphRecommendationLayer:
	def __init__(self, foundation, users, items, interactions):
		self.engine = GraphRecommendationBuilder(foundation, users, items, interactions)

	def generate(self):
		return self.engine.generate()
