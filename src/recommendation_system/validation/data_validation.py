from __future__ import annotations

import pandas as pd


class DataValidator:
	def validate_required_columns(self, frame: pd.DataFrame, required: set[str]) -> None:
		missing = required.difference(frame.columns)
		if missing:
			raise ValueError(f"missing columns: {sorted(missing)}")

	def validate(self, frame: pd.DataFrame, required: set[str] | None = None) -> bool:
		if frame is None or frame.empty:
			raise ValueError("dataframe must be non-empty")
		if required:
			self.validate_required_columns(frame, required)
		if frame.isna().any().any():
			raise ValueError("missing values detected")
		return True
