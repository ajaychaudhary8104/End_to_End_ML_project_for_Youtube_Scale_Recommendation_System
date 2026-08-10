from __future__ import annotations

import numpy as np
import pandas as pd

from .foundation_builder import FoundationLayer


class ArchetypeEngine:
	DEFAULT_ARCHETYPES = [
		"Explorer",
		"Loyalist",
		"Passive Consumer",
		"Heavy Consumer",
		"Trend Seeker",
		"Niche Consumer",
		"Social Influenced",
		"Seasonal User",
		"High Value User",
		"Churn Risk User",
	]

	def __init__(self, foundation: FoundationLayer):
		self.foundation = foundation
		self.rng = foundation.rng

	def generate(self) -> pd.DataFrame:
		count = self.rng.integers(
			self.foundation.config.min_archetypes,
			self.foundation.config.max_archetypes + 1,
		)
		selected = self.DEFAULT_ARCHETYPES[:count]
		return pd.DataFrame(
			{
				"archetype_id": np.arange(1, len(selected) + 1),
				"archetype_name": selected,
				"retention_score": self.rng.uniform(0.20, 1.00, len(selected)),
				"churn_probability": self.rng.uniform(0.01, 0.60, len(selected)),
				"average_ltv": self.rng.uniform(100, 5000, len(selected)),
				"content_diversity": self.rng.uniform(0.10, 1.00, len(selected)),
			}
		)
