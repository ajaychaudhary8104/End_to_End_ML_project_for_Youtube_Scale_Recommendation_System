from __future__ import annotations

import numpy as np
import pandas as pd

from ..foundation import FoundationLayer


class PositionBiasEngine:
	def __init__(self, foundation: FoundationLayer, contexts: pd.DataFrame | None = None):
		self.foundation = foundation
		self.rng = foundation.rng
		self.contexts = contexts

	def _coerce(self, contexts: pd.DataFrame) -> pd.DataFrame:
		if contexts is None or contexts.empty:
			raise ValueError("contexts must be a non-empty DataFrame")
		result = contexts.copy()
		if "page_position" not in result:
			result["page_position"] = self.rng.integers(1, 101, len(result))
		if "recommendation_slot" not in result:
			result["recommendation_slot"] = self.rng.integers(1, 51, len(result))
		if "surface_type" not in result:
			result["surface_type"] = self.rng.choice(["homepage", "search", "recommended"], len(result))
		return result

	def generate(self, contexts: pd.DataFrame | None = None) -> pd.DataFrame:
		result = self._coerce(contexts if contexts is not None else self.contexts)
		positions = result["page_position"].clip(lower=1).to_numpy(dtype=float)
		decay = self.rng.uniform(0.55, 1.30, len(result))
		raw = np.power(positions, -decay)
		position = raw / max(raw.max(), 1e-12)
		visibility = np.clip(1 - (positions - 1) / max(positions.max(), 1) * 0.90, 0.05, 1.0)
		surface = result["surface_type"].to_numpy()
		visibility[surface == "homepage"] += 0.05
		visibility[surface == "search"] -= 0.03
		visibility[surface == "recommended"] += 0.02
		visibility = np.clip(visibility, 0.05, 1.0)
		surface_bias = result.get("homepage_bias", pd.Series(0.0, index=result.index)).to_numpy()
		search_bias = result.get("search_bias", pd.Series(0.0, index=result.index)).to_numpy()
		exposure = np.clip(0.55 * position + 0.35 * visibility + 0.10 * np.where(surface == "search", search_bias, surface_bias), 0, 1)
		output = pd.DataFrame({"position_bias_score": position, "visibility_score": visibility, "exposure_probability": exposure}, index=result.index)
		if "context_id" in result:
			output.insert(0, "context_id", result["context_id"].to_numpy())
		return output


class Phase6PositionBiasLayer:
	def __init__(self, foundation, contexts):
		self.engine = PositionBiasEngine(foundation, contexts)
		self.contexts = contexts

	def generate(self):
		return self.engine.generate(self.contexts)
