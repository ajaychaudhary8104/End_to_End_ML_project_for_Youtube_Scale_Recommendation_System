from __future__ import annotations

import pandas as pd


class TrainingDatasetBuilder:
	def build_retrieval(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
		columns = [column for column in ["user_id", "item_id", "candidate_generation_score", "positive_interaction_label"] if column in affinity_df]
		return affinity_df[columns].copy()

	def build_ranking(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
		columns = [column for column in ["user_id", "item_id", "ranking_label"] if column in affinity_df]
		return affinity_df[columns].copy()

	def build_bandit(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
		columns = [column for column in ["user_id", "item_id", "bandit_reward"] if column in affinity_df]
		return affinity_df[columns].copy()

	def generate(self, affinity_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
		if affinity_df is None or affinity_df.empty:
			raise ValueError("affinity_df must be a non-empty DataFrame")
		return {
			"retrieval_dataset": self.build_retrieval(affinity_df),
			"ranking_dataset": self.build_ranking(affinity_df),
			"bandit_dataset": self.build_bandit(affinity_df),
		}
