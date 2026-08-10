from __future__ import annotations

import pandas as pd


class ContextFeatureBuilder:
	def generate(self, contexts: pd.DataFrame) -> pd.DataFrame:
		if contexts is None or contexts.empty:
			raise ValueError("contexts must be a non-empty DataFrame")
		result = contexts.copy()
		if "hour_of_day" in result:
			result["is_prime_time"] = result["hour_of_day"].between(18, 22)
		if "day_of_week" in result:
			result["is_weekend"] = result["day_of_week"].isin([5, 6])
		return result

	build = generate
