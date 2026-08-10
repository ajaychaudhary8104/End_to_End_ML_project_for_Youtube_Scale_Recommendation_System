from __future__ import annotations

import pandas as pd


class QualityChecker:
	def check_unique(self, frame: pd.DataFrame, column: str) -> bool:
		if column not in frame or frame[column].duplicated().any():
			raise ValueError(f"duplicate or missing key column: {column}")
		return True

	def check_range(self, frame: pd.DataFrame, columns: list[str], lower: float = 0.0, upper: float = 1.0) -> bool:
		for column in columns:
			if column in frame and not frame[column].between(lower, upper).all():
				raise ValueError(f"invalid range in {column}")
		return True

	def run(self, frame: pd.DataFrame, key: str | None = None, score_columns: list[str] | None = None) -> bool:
		if frame is None or frame.empty:
			raise ValueError("dataframe must be non-empty")
		if key:
			self.check_unique(frame, key)
		if score_columns:
			self.check_range(frame, score_columns)
		return True
