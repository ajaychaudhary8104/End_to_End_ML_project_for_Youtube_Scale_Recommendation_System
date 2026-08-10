from __future__ import annotations

import numpy as np
import pandas as pd


class ReRankingGenerator:
	def __init__(self, foundation, ranking_df=None, items=None):
		self.foundation, self.rng = foundation, foundation.rng
		self.ranking_df, self.items = ranking_df, items

	def generate(self, ranking_df=None, items=None):
		ranking_df = ranking_df if ranking_df is not None else self.ranking_df
		items = items if items is not None else self.items
		if ranking_df is None or items is None:
			raise ValueError("ranking_df and items must be provided")
		item_lookup = items.set_index("item_id")
		output = ranking_df.copy()
		diversity, novelty = [], []
		for _, row in output.iterrows():
			query = output[output["query_id"] == row["query_id"]]
			genres = [item_lookup.loc[item_id].get("genre", "unknown") for item_id in query["item_id"] if item_id in item_lookup.index]
			diversity.append(min(1.0, len(set(genres)) / max(len(query), 1)))
			novelty.append(float(item_lookup.loc[row["item_id"]].get("freshness_score", 0.5)))
		output["diversity_score"] = np.clip(diversity, 0, 1)
		output["novelty_score"] = np.clip(novelty, 0, 1)
		output["rerank_score"] = np.clip(output["rank_label"] * 0.45 + output["diversity_score"] * 0.35 + output["novelty_score"] * 0.20, 0, 1)
		return output


class Phase15ReRankingLayer:
	def __init__(self, foundation, ranking_df, items):
		self.engine = ReRankingGenerator(foundation, ranking_df, items)

	def generate(self):
		return self.engine.generate()
