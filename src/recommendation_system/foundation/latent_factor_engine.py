from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .foundation_builder import FoundationLayer


class LatentFactorEngine:
	def __init__(self, foundation: FoundationLayer):
		self.foundation = foundation
		self.rng = foundation.rng

	def _generate(self, count: int) -> np.ndarray:
		embeddings = self.rng.normal(
			loc=0.0,
			scale=1.0,
			size=(count, self.foundation.config.embedding_dim),
		)
		if self.foundation.config.normalize_embeddings:
			embeddings = self.foundation.normalize_vectors(embeddings)
		return embeddings

	def generate_user_embeddings(self) -> np.ndarray:
		return self._generate(self.foundation.config.n_users)

	def generate_item_embeddings(self) -> np.ndarray:
		return self._generate(self.foundation.config.n_items)

	def build_embedding_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
		user_df = pd.DataFrame(self.generate_user_embeddings())
		user_df.insert(0, "user_id", np.arange(1, len(user_df) + 1))

		item_df = pd.DataFrame(self.generate_item_embeddings())
		item_df.insert(0, "item_id", np.arange(1, len(item_df) + 1))
		return user_df, item_df
