from __future__ import annotations

import numpy as np
import pandas as pd


class RLDataSetBuilder:
	def __init__(self, foundation, users=None, items=None, interactions=None, affinity_df=None):
		self.foundation, self.rng = foundation, foundation.rng
		self.users, self.items, self.interactions, self.affinity_df = users, items, interactions, affinity_df

	def generate(self, users=None, items=None, interactions=None, affinity_df=None) -> pd.DataFrame:
		interactions = interactions if interactions is not None else self.interactions
		affinity_df = affinity_df if affinity_df is not None else self.affinity_df
		if interactions is None or interactions.empty:
			raise ValueError("interactions must be provided")
		result = pd.DataFrame({"user_id": interactions["user_id"].to_numpy(), "action_item_id": interactions["item_id"].to_numpy()})
		clicked = interactions.get("click", interactions.get("clicked", pd.Series(np.zeros(len(interactions)))))
		watched = interactions.get("watch", interactions.get("watched", pd.Series(np.zeros(len(interactions)))))
		completed = interactions.get("completion", interactions.get("completed", pd.Series(np.zeros(len(interactions)))))
		result["reward"] = clicked.astype(int).to_numpy() + watched.astype(int).to_numpy() + completed.astype(int).to_numpy()
		if affinity_df is not None and "bandit_reward" in affinity_df:
			keys = affinity_df.set_index(["user_id", "item_id"])["bandit_reward"]
			result["reward"] = [float(keys.get((user_id, item_id), reward)) for user_id, item_id, reward in zip(result["user_id"], result["action_item_id"], result["reward"])]
		result["done"] = completed.astype(bool).to_numpy()
		result["state_index"] = np.arange(len(result), dtype=np.int64)
		result["next_state_index"] = np.minimum(result["state_index"] + 1, max(len(result) - 1, 0))
		return result

	build_rl_dataset = generate


class ReinforcementLearningDatasetBuilder(RLDataSetBuilder):
	pass
