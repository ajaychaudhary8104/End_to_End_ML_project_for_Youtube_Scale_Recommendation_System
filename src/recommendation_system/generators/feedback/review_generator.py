from __future__ import annotations

import numpy as np
import pandas as pd

from ...foundation import FoundationLayer


class ReviewGenerator:
	TEMPLATES = {
		"positive": ["Loved the experience and would watch again.", "Great recommendation with a smooth and engaging flow."],
		"neutral": ["A decent watch overall with some enjoyable moments.", "It was fine, not memorable but still acceptable."],
		"negative": ["The recommendation did not fit my interests well.", "It was slow and not very engaging."],
	}

	def __init__(self, foundation: FoundationLayer, users=None, items=None, interactions=None):
		self.foundation = foundation
		self.rng = foundation.rng
		self.users, self.items, self.interactions = users, items, interactions

	def generate(self, users=None, items=None, interactions=None) -> pd.DataFrame:
		interactions = interactions if interactions is not None else self.interactions
		if interactions is None or interactions.empty:
			raise ValueError("interactions must be provided")
		rows = []
		for _, row in interactions.iterrows():
			rating = float(np.round(self.rng.uniform(3, 5) if row.get("watch", 0) else self.rng.uniform(1, 3.5), 1))
			sentiment = "positive" if rating >= 4 else ("negative" if rating <= 2 else "neutral")
			rows.append({"user_id": int(row["user_id"]), "item_id": int(row["item_id"]), "rating": rating, "review_text": self.rng.choice(self.TEMPLATES[sentiment]), "sentiment": sentiment, "helpfulness": float(self.rng.beta(5, 2) if sentiment == "positive" else self.rng.beta(2, 5) if sentiment == "negative" else self.rng.beta(3, 3))})
		return pd.DataFrame(rows)


class Phase12ReviewLayer:
	def __init__(self, foundation, users, items, interactions):
		self.engine = ReviewGenerator(foundation, users, items, interactions)

	def generate(self):
		return self.engine.generate()
