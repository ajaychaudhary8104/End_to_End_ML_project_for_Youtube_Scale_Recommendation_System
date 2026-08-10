from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..foundation import FoundationLayer


class AffinityEngine:
	"""Generate user-item affinity, propensity, and training labels."""

	def __init__(
		self,
		foundation: FoundationLayer,
		users: pd.DataFrame,
		items: pd.DataFrame,
		user_embeddings: pd.DataFrame | np.ndarray | None = None,
		item_embeddings: pd.DataFrame | np.ndarray | None = None,
	):
		self.foundation = foundation
		self.config = foundation.config
		self.rng = foundation.rng
		self.users = users.reset_index(drop=True)
		self.items = items.reset_index(drop=True)
		self.user_embeddings = self._embedding_array(user_embeddings, "user_id")
		self.items_content_embeddings = self._embedding_array(item_embeddings, "item_id")
		if self.items_content_embeddings is None:
			self.items_content_embeddings = self._fallback_embeddings(len(self.items), self.config.embedding_dim)
		if self.user_embeddings is None:
			self.user_embeddings = self._fallback_embeddings(len(self.users), self.config.embedding_dim)
		self._validate_inputs()

	@staticmethod
	def _embedding_array(embeddings, id_column: str) -> np.ndarray | None:
		if embeddings is None:
			return None
		if isinstance(embeddings, pd.DataFrame):
			values = embeddings.drop(columns=[id_column], errors="ignore").to_numpy(dtype=np.float32)
		else:
			values = np.asarray(embeddings, dtype=np.float32)
		if values.ndim != 2:
			raise ValueError("embeddings must be a two-dimensional array")
		norms = np.linalg.norm(values, axis=1, keepdims=True)
		return values / np.maximum(norms, 1e-12)

	def _fallback_embeddings(self, rows: int, dimensions: int) -> np.ndarray:
		values = self.rng.normal(0, 1, (rows, dimensions))
		return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

	def _validate_inputs(self) -> None:
		for table, name, required in (
			(self.users, "users", {"user_id", "preferred_genres", "preferred_language", "popularity_preference", "novelty_preference", "primary_device", "age"}),
			(self.items, "items", {"item_id", "sub_genres", "language", "popularity_score", "freshness_score", "quality_score", "creator", "maturity_rating", "completion_rate", "retention_impact"}),
		):
			missing = required.difference(table.columns)
			if missing:
				raise ValueError(f"{name} missing columns: {sorted(missing)}")

	def _genre_affinity(self, user: pd.Series, item: pd.Series) -> float:
		return len(set(user["preferred_genres"]) & set(item["sub_genres"])) / max(len(item["sub_genres"]), 1)

	def _language_affinity(self, user: pd.Series, item: pd.Series) -> float:
		return float(user["preferred_language"] == item["language"])

	def _popularity_affinity(self, user: pd.Series, item: pd.Series) -> float:
		return float(1 - abs(user["popularity_preference"] - item["popularity_score"]))

	def _novelty_affinity(self, user: pd.Series, item: pd.Series) -> float:
		return float(1 - abs(item["freshness_score"] - user["novelty_preference"]))

	def _quality_affinity(self, item: pd.Series) -> float:
		return float(item["quality_score"])

	def _recency_affinity(self, item: pd.Series) -> float:
		return float(item["freshness_score"])

	def _creator_affinity(self, user: pd.Series, item: pd.Series) -> float:
		digest = hashlib.sha256(str(item["creator"]).encode("utf-8")).digest()
		if int.from_bytes(digest[:2], "big") % 100 < 15:
			return float(self.rng.uniform(0.7, 1.0))
		return float(self.rng.uniform(0.0, 0.6))

	def _user_item_embedding_affinity(self, user: pd.Series, item: pd.Series) -> float:
		user_index = int(user.name) % len(self.user_embeddings)
		item_index = int(item.name) % len(self.items_content_embeddings)
		similarity = float(np.dot(self.user_embeddings[user_index], self.items_content_embeddings[item_index]))
		return float(np.clip((similarity + 1.0) / 2.0, 0, 1))

	def _device_affinity(self, user: pd.Series, context: pd.Series) -> float:
		return 1.0 if user["primary_device"] == context["device_type"] else 0.50

	def _time_affinity(self, user: pd.Series, context: pd.Series) -> float:
		preferred_hour = user.get("preferred_hour", 20 if user["age"] >= 25 else 21)
		distance = min(abs(float(preferred_hour) - context["hour_of_day"]), 24 - abs(float(preferred_hour) - context["hour_of_day"]))
		return float(np.clip(1 - distance / 12, 0, 1))

	def _seasonality_affinity(self, user: pd.Series, context: pd.Series) -> float:
		return float(user.get("preferred_season", context["season"]) == context["season"])

	def _maturity_affinity(self, user: pd.Series, item: pd.Series) -> float:
		preference = user.get("content_maturity_preference", "Adult")
		levels = {"Kids": 0.1, "Teen": 0.5, "Adult": 0.8, "G": 0.1, "PG": 0.3, "PG13": 0.5, "R": 0.8, "NC17": 1.0}
		return float(1 - abs(levels.get(preference, 0.8) - levels.get(item["maturity_rating"], 0.5)))

	def _context_affinity(self, context: pd.Series) -> float:
		return float(np.clip(context["watch_intent_score"] * 0.40 + context["attention_level"] * 0.20 + context["network_quality_score"] * 0.15 + context["context_quality_score"] * 0.25, 0, 1))

	def _personalization_score(self, user: pd.Series, item: pd.Series, context: pd.Series) -> float:
		values = [
			self._genre_affinity(user, item),
			self._language_affinity(user, item),
			self._maturity_affinity(user, item),
			self._device_affinity(user, context),
			self._time_affinity(user, context),
		]
		return float(np.mean(values))

	def calculate_affinity(self, user: pd.Series, item: pd.Series, context: pd.Series) -> dict[str, float]:
		metadata = (
			self._genre_affinity(user, item) * 0.30
			+ self._creator_affinity(user, item) * 0.10
			+ self._language_affinity(user, item) * 0.10
			+ self._popularity_affinity(user, item) * 0.15
			+ self._novelty_affinity(user, item) * 0.10
			+ self._quality_affinity(item) * 0.15
			+ self._recency_affinity(item) * 0.10
		)
		embedding = self._user_item_embedding_affinity(user, item)
		contextual = self._context_affinity(context)
		device = self._device_affinity(user, context)
		time = self._time_affinity(user, context)
		season = self._seasonality_affinity(user, context)
		personalization = self._personalization_score(user, item, context)
		score = np.clip(metadata * 0.35 + embedding * 0.25 + contextual * 0.10 + device * 0.05 + time * 0.05 + season * 0.05 + personalization * 0.15 + self.rng.normal(0, 0.02), 0, 1)
		return {
			"affinity_score": float(score),
			"metadata_affinity": float(metadata),
			"embedding_affinity": float(embedding),
			"context_affinity": float(contextual),
			"device_affinity": float(device),
			"time_affinity": float(time),
			"seasonality_affinity": float(season),
			"personalization_score": float(personalization),
		}

	def _click_probability(self, score: float, context: pd.Series) -> float:
		return float(np.clip(score * 0.70 + context["homepage_bias"] * 0.15 + context["watch_intent_score"] * 0.15 + self.rng.normal(0, 0.03), 0, 1))

	def _watch_probability(self, score: float, click: float) -> float:
		return float(np.clip(score * 0.50 + click * 0.50, 0, 1))

	def _completion_probability(self, watch: float, item: pd.Series) -> float:
		return float(np.clip(watch * 0.70 + item["completion_rate"] * 0.30, 0, 1))

	def _satisfaction_probability(self, score: float, completion: float) -> float:
		return float(np.clip(score * 0.50 + completion * 0.50, 0, 1))

	def _retention_probability(self, satisfaction: float, item: pd.Series) -> float:
		return float(np.clip(satisfaction * 0.70 + item["retention_impact"] * 0.30, 0, 1))

	def sample_candidate_items(self, user: pd.Series, candidates_per_user: int) -> np.ndarray:
		count = min(max(int(candidates_per_user), 1), len(self.items))
		weights = self.items["retrieval_score"].to_numpy(dtype=float) if "retrieval_score" in self.items else np.ones(len(self.items))
		weights = np.maximum(weights, 1e-8) / np.maximum(weights.sum(), 1e-8)
		return self.rng.choice(len(self.items), size=count, replace=False, p=weights)

	def enrich_affinity_dataset(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
		result = affinity_df.copy()
		result["affinity_embedding_id"] = np.arange(1, len(result) + 1, dtype=np.int64)
		result["affinity_cluster"] = np.digitize(result["affinity_score"], [0.20, 0.40, 0.60, 0.80]).astype(np.int32)
		result["candidate_generation_score"] = np.clip(result["affinity_score"] * 0.50 + result["watch_probability"] * 0.30 + result["click_probability"] * 0.20, 0, 1)
		result["ranking_label"] = np.clip(result["watch_probability"] * 0.40 + result["completion_probability"] * 0.30 + result["satisfaction_probability"] * 0.30, 0, 1)
		result["bandit_reward"] = np.clip(result["retention_probability"] * 0.50 + result["satisfaction_probability"] * 0.50, 0, 1)
		result["positive_interaction_label"] = (result["click_probability"] >= 0.50).astype(np.int8)
		result["high_value_label"] = (result["retention_probability"] >= 0.75).astype(np.int8)
		return result

	def validate_affinity_dataset(self, affinity_df: pd.DataFrame) -> None:
		required = {"user_id", "item_id", "affinity_score", "click_probability", "watch_probability", "completion_probability", "retention_probability"}
		missing = required.difference(affinity_df.columns)
		if missing:
			raise ValueError(f"missing affinity columns: {sorted(missing)}")
		if affinity_df.empty or affinity_df[["user_id", "item_id"]].duplicated().any():
			raise ValueError("invalid or duplicate affinity pairs")
		for column in [column for column in affinity_df.columns if column.endswith("probability") or column.endswith("_affinity") or column in {"affinity_score", "candidate_generation_score", "ranking_label", "bandit_reward"}]:
			if affinity_df[column].isna().any() or not affinity_df[column].between(0, 1).all():
				raise ValueError(f"invalid values in {column}")

	def generate_affinity_dataset(self, contexts: pd.DataFrame, candidates_per_user: int = 500) -> pd.DataFrame:
		if contexts.empty:
			raise ValueError("contexts must not be empty")
		rows = []
		for _, user in self.users.iterrows():
			context = contexts.iloc[int(self.rng.integers(0, len(contexts)))]
			for item_index in self.sample_candidate_items(user, candidates_per_user):
				item = self.items.iloc[item_index]
				values = self.calculate_affinity(user, item, context)
				click = self._click_probability(values["affinity_score"], context)
				watch = self._watch_probability(values["affinity_score"], click)
				completion = self._completion_probability(watch, item)
				satisfaction = self._satisfaction_probability(values["affinity_score"], completion)
				retention = self._retention_probability(satisfaction, item)
				rows.append({"user_id": user["user_id"], "item_id": item["item_id"], "context_id": context["context_id"], **values, "click_probability": click, "watch_probability": watch, "completion_probability": completion, "satisfaction_probability": satisfaction, "retention_probability": retention, "churn_reduction_probability": retention * 0.95})
		result = self.enrich_affinity_dataset(pd.DataFrame(rows))
		self.validate_affinity_dataset(result)
		return result

	def build_user_item_score_matrix(self, affinity_df: pd.DataFrame):
		from scipy.sparse import csr_matrix
		user_domain = pd.Index(self.users["user_id"].unique())
		item_domain = pd.Index(self.items["item_id"].unique())
		user_codes = pd.Categorical(affinity_df["user_id"], categories=user_domain).codes
		item_codes = pd.Categorical(affinity_df["item_id"], categories=item_domain).codes
		return csr_matrix(
			(affinity_df["affinity_score"], (user_codes, item_codes)),
			shape=(len(user_domain), len(item_domain)),
		)

	def build_affinity_feature_store(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
		result = affinity_df.copy()
		result["event_timestamp"] = pd.Timestamp.utcnow()
		return result

	def build_two_tower_dataset(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
		return affinity_df[["user_id", "item_id", "candidate_generation_score", "positive_interaction_label"]].copy()

	def build_ranking_dataset(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
		return affinity_df[["user_id", "item_id", "ranking_label"]].copy()

	def build_bandit_dataset(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
		return affinity_df[["user_id", "item_id", "bandit_reward"]].copy()

	def build_serving_affinity_features(self, affinity_df: pd.DataFrame) -> pd.DataFrame:
		return affinity_df[["user_id", "item_id", "affinity_score", "candidate_generation_score", "ranking_label", "bandit_reward", "watch_probability", "click_probability", "retention_probability"]].copy()
