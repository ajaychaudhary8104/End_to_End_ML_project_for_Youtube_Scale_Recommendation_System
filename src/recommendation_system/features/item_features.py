from __future__ import annotations

import pandas as pd


class ItemFeatureBuilder:
	def generate(self, items: pd.DataFrame) -> pd.DataFrame:
		if items is None or items.empty:
			raise ValueError("items must be a non-empty DataFrame")
		result = items.copy()
		if "release_date" in result:
			result["release_year"] = pd.to_datetime(result["release_date"]).dt.year
		for column in ["quality_score", "popularity_score", "trend_score", "freshness_score"]:
			if column in result:
				result[f"{column}_rank"] = result[column].rank(pct=True)
		return result

	build = generate
