from __future__ import annotations

import pandas as pd


class UserFeatureBuilder:
	def generate(self, users: pd.DataFrame) -> pd.DataFrame:
		if users is None or users.empty:
			raise ValueError("users must be a non-empty DataFrame")
		result = users.copy()
		for column in result.select_dtypes(include="number").columns:
			result[f"{column}_missing"] = result[column].isna().astype("int8")
		return result

	build = generate
