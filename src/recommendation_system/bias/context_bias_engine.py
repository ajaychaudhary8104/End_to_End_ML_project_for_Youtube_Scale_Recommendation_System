from __future__ import annotations

import numpy as np
import pandas as pd

from ..foundation import FoundationLayer


class ContextBiasEngine:
	def __init__(self, foundation: FoundationLayer, contexts: pd.DataFrame | None = None, affinity_df: pd.DataFrame | None = None):
		self.foundation = foundation
		self.rng = foundation.rng
		self.contexts = contexts
		self.affinity_df = affinity_df

	def _coerce_contexts(self, contexts: pd.DataFrame) -> pd.DataFrame:
		if contexts is None or contexts.empty:
			raise ValueError("contexts must be a non-empty DataFrame")
		result = contexts.copy()
		defaults = {
			"context_id": np.arange(1, len(result) + 1),
			"device_type": self.rng.choice(["Mobile", "Desktop", "Tablet", "SmartTV"], len(result)),
			"hour_of_day": self.rng.integers(0, 24, len(result)),
			"day_of_week": self.rng.integers(0, 7, len(result)),
			"month": self.rng.integers(1, 13, len(result)),
			"season": self.rng.choice(["winter", "spring", "summer", "autumn"], len(result)),
			"traffic_source": self.rng.choice(["organic", "push_notification", "email", "advertisement"], len(result)),
		}
		for column, values in defaults.items():
			if column not in result:
				result[column] = values
		return result

	def generate(self, contexts=None, affinity_df=None):
		contexts = self._coerce_contexts(contexts if contexts is not None else self.contexts)
		affinity_df = affinity_df if affinity_df is not None else self.affinity_df
		if affinity_df is None or affinity_df.empty:
			raise ValueError("affinity_df must be a non-empty DataFrame")
		affinity = affinity_df.copy()
		if "context_id" not in affinity:
			affinity["context_id"] = self.rng.choice(contexts["context_id"].to_numpy(), len(affinity))
		merged = affinity.merge(contexts, on="context_id", how="left", suffixes=("", "_context"))
		weekend = np.where(merged["day_of_week"].isin([5, 6]), self.rng.uniform(0.10, 0.35, len(merged)), 0)
		holiday = np.where(merged["month"].isin([11, 12, 1]), self.rng.uniform(0.05, 0.30, len(merged)), 0)
		mobile = np.where(merged["device_type"] == "Mobile", self.rng.uniform(-0.12, 0.08, len(merged)), 0)
		prime = np.where(merged["hour_of_day"].between(18, 22), self.rng.uniform(0.08, 0.28, len(merged)), 0)
		campaign = np.where(merged["traffic_source"].isin(["push_notification", "email", "advertisement"]), self.rng.uniform(0.05, 0.22, len(merged)), 0)
		bias = np.clip(weekend + holiday + mobile + prime + campaign, -0.25, 0.55)
		signal_column = next((column for column in ["adjusted_affinity", "candidate_generation_score", "ranking_label", "bandit_reward", "affinity_score"] if column in merged), None)
		base = merged[signal_column].to_numpy(float) if signal_column else np.full(len(merged), 0.5)
		merged["context_bias"] = bias
		merged["adjusted_affinity"] = np.clip(base + bias, 0, 1)
		return merged


class Phase7ContextBiasLayer:
	def __init__(self, foundation, contexts, affinity_df):
		self.engine = ContextBiasEngine(foundation, contexts, affinity_df)

	def generate(self):
		return self.engine.generate()
