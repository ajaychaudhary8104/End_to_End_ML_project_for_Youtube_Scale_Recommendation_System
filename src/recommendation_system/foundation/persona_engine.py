from __future__ import annotations

import numpy as np
import pandas as pd

from .foundation_builder import FoundationLayer


class PersonaEngine:
	DEFAULT_PERSONAS = [
		"Casual Viewer",
		"Binge Watcher",
		"Weekend User",
		"Sports Fan",
		"Movie Enthusiast",
		"Documentary Lover",
		"Anime Fan",
		"Family User",
		"Kids User",
		"Premium Power User",
		"Trend Chaser",
		"News Consumer",
		"Music Addict",
		"Mobile User",
		"Creator Follower",
	]

	def __init__(self, foundation: FoundationLayer):
		self.foundation = foundation
		self.rng = foundation.rng

	def generate(self) -> pd.DataFrame:
		count = self.rng.integers(
			self.foundation.config.min_personas,
			self.foundation.config.max_personas + 1,
		)
		selected = self.DEFAULT_PERSONAS[:count]
		return pd.DataFrame(
			{
				"persona_id": np.arange(1, len(selected) + 1),
				"persona_name": selected,
				"avg_session_length": self.rng.uniform(10, 180, len(selected)),
				"daily_active_probability": self.rng.uniform(
					0.05, 0.95, len(selected)
				),
				"exploration_rate": self.rng.uniform(0.01, 0.50, len(selected)),
				"engagement_score": self.rng.uniform(0.20, 1.00, len(selected)),
			}
		)
