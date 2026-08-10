from __future__ import annotations

import numpy as np
import pandas as pd

from ...foundation import FoundationLayer


class ItemGenerator:
	"""Generate catalog metadata, lifecycle signals, and item features."""

	CONTENT_TYPES = ["Movie", "Series", "Short", "Documentary", "Live", "Podcast"]
	GENRES = [
		"Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror",
		"Mystery", "Romance", "SciFi", "Thriller", "Documentary",
		"Animation", "Sports", "News", "Music",
	]
	LANGUAGES = ["English", "Hindi", "German", "Japanese", "Portuguese", "Spanish", "French"]
	COUNTRIES = ["USA", "India", "UK", "Germany", "Japan", "Brazil", "Canada", "Australia"]
	MATURITY_RATINGS = ["Kids", "Teen", "Adult"]
	STUDIOS = [f"Studio_{letter}" for letter in "ABCDEFG"]

	def __init__(self, foundation: FoundationLayer, item_embeddings: pd.DataFrame | None = None):
		self.foundation = foundation
		self.config = foundation.config
		self.rng = foundation.rng
		self.item_embeddings = item_embeddings
		self.content_embedding_matrix: np.ndarray | None = None
		self.genre_embedding_matrix: np.ndarray | None = None
		self.creator_embedding_matrix: np.ndarray | None = None

	def _generate_genres(self, count: int) -> tuple[np.ndarray, list[list[str]]]:
		primary, subgenres = [], []
		for _ in range(count):
			genres = list(self.rng.choice(self.GENRES, self.rng.integers(1, 4), replace=False))
			primary.append(genres[0])
			subgenres.append(genres)
		return np.asarray(primary), subgenres

	def _generate_runtime(self, content_types: np.ndarray) -> np.ndarray:
		bounds = {
			"Movie": (80, 180), "Series": (20, 60), "Short": (3, 20),
			"Documentary": (10, 120), "Live": (10, 120), "Podcast": (10, 120),
		}
		return np.asarray([self.rng.integers(*bounds[item_type]) for item_type in content_types])

	def generate_base_items(self) -> pd.DataFrame:
		count = self.config.n_items
		release_start = pd.Timestamp("2010-01-01")
		release_end = pd.Timestamp(self.config.end_date)
		days = max((release_end - release_start).days, 1)
		release_dates = pd.Series(
			release_start + pd.to_timedelta(self.rng.integers(0, days, count), unit="D")
		)
		genres, subgenres = self._generate_genres(count)
		content_types = self.rng.choice(self.CONTENT_TYPES, count, p=[0.35, 0.30, 0.10, 0.10, 0.05, 0.10])
		return pd.DataFrame(
			{
				"item_id": np.arange(1, count + 1),
				"title": [f"Content_{index:07d}" for index in range(1, count + 1)],
				"content_type": content_types,
				"genre": genres,
				"sub_genres": subgenres,
				"language": self.rng.choice(self.LANGUAGES, count),
				"country": self.rng.choice(self.COUNTRIES, count),
				"release_date": release_dates,
				"release_year": release_dates.dt.year,
				"runtime_minutes": self._generate_runtime(content_types),
				"maturity_rating": self.rng.choice(self.MATURITY_RATINGS, count, p=[0.15, 0.30, 0.55]),
				"creator": [f"Creator_{value}" for value in self.rng.integers(1, 5000, count)],
				"studio": self.rng.choice(self.STUDIOS, count),
			}
		)

	def enrich_quality_features(self, items: pd.DataFrame) -> pd.DataFrame:
		items = items.copy()
		popularity = self.rng.beta(2.5, 5.0, len(items))
		blockbuster = self.rng.random(len(items)) < 0.05
		popularity[blockbuster] = np.clip(popularity[blockbuster] + self.rng.uniform(0.30, 0.60, blockbuster.sum()), 0, 1)
		quality = self.rng.beta(4.0, 2.0, len(items))
		critic = np.clip(quality * 100 + self.rng.normal(0, 8, len(items)), 0, 100)
		audience = np.clip((popularity * 0.40 + quality * 0.60) * 100 + self.rng.normal(0, 10, len(items)), 0, 100)
		virality = self.rng.beta(1.5, 8.0, len(items))
		viral = self.rng.random(len(items)) < 0.02
		virality[viral] = self.rng.uniform(0.85, 1.0, viral.sum())
		age = (pd.Timestamp(self.config.end_date) - items["release_date"]).dt.days.to_numpy()
		freshness = np.clip(np.exp(-age / 365), 0, 1)
		trend = np.clip(freshness * 0.70 + virality * 0.30 + self.rng.normal(0, 0.05, len(items)), 0, 1)
		evergreen = np.clip(quality + self.rng.normal(0, 0.05, len(items)), 0, 1)
		completion = np.clip(0.35 + quality * 0.60 + self.rng.normal(0, 0.05, len(items)), 0, 1)
		retention = np.clip(quality * 0.60 + evergreen * 0.40, 0, 1)
		items["popularity_score"] = popularity
		items["quality_score"] = quality
		items["critic_score"] = critic
		items["audience_score"] = audience
		items["virality_score"] = virality
		items["trend_score"] = trend
		items["evergreen_score"] = evergreen
		items["completion_rate"] = completion
		items["retention_impact"] = retention
		items["business_value_score"] = np.clip(popularity * 0.40 + quality * 0.30 + retention * 0.30, 0, 1)
		return items

	def enrich_lifecycle_features(self, items: pd.DataFrame) -> pd.DataFrame:
		items = items.copy()
		age = (pd.Timestamp(self.config.end_date) - items["release_date"]).dt.days.to_numpy().astype(np.int32)
		status = self.rng.choice(["active", "licensed", "expiring", "retired"], len(items), p=[0.78, 0.15, 0.05, 0.02])
		risk = np.zeros(len(items))
		for state, low, high in [("active", 0.00, 0.20), ("licensed", 0.20, 0.60), ("expiring", 0.60, 0.95)]:
			mask = status == state
			risk[mask] = self.rng.uniform(low, high, mask.sum())
		risk[status == "retired"] = 1.0
		stage = np.select([age <= 30, age <= 180, age <= 730], ["launch", "growth", "mature"], default="decline")
		items["content_age_days"] = age
		items["freshness_score"] = np.clip(np.exp(-age / 365), 0, 1)
		items["lifecycle_stage"] = stage
		items["catalog_status"] = status
		items["seasonal_content_flag"] = items["genre"].isin({"Sports", "Music", "News"})
		items["licensing_risk"] = risk
		items["availability_status"] = np.select([risk >= 1.0, risk > 0.75], ["removed", "restricted"], default="available")
		segment = np.full(len(items), "standard", dtype=object)
		segment[items["popularity_score"].to_numpy() > 0.80] = "blockbuster"
		segment[items["quality_score"].to_numpy() > 0.90] = "prestige"
		segment[(items["popularity_score"].to_numpy() < 0.20) & (items["quality_score"].to_numpy() > 0.75)] = "hidden_gem"
		segment[items["trend_score"].to_numpy() > 0.80] = "trending"
		items["catalog_segment"] = segment
		items["launch_momentum"] = np.clip(items["freshness_score"] * 0.60 + items["virality_score"] * 0.40, 0, 1)
		items["long_term_value"] = np.clip(items["quality_score"] * 0.50 + items["evergreen_score"] * 0.50, 0, 1)
		return items

	def enrich_recommendation_features(self, items: pd.DataFrame) -> pd.DataFrame:
		items = items.copy()
		popularity = items["popularity_score"].to_numpy()
		quality = items["quality_score"].to_numpy()
		trend = items["trend_score"].to_numpy()
		items["item_embedding_id"] = items["item_id"]
		items["embedding_cluster"] = self.rng.integers(0, 100, len(items))
		items["retrieval_score"] = np.clip(popularity * 0.35 + trend * 0.35 + quality * 0.30, 0, 1)
		items["ranking_score"] = np.clip(quality * 0.40 + items["completion_rate"] * 0.25 + items["audience_score"] / 100 * 0.20 + trend * 0.15, 0, 1)
		items["diversity_score"] = np.clip(1 - popularity + self.rng.normal(0, 0.05, len(items)), 0, 1)
		items["exploration_score"] = np.clip(items["freshness_score"] * 0.50 + items["diversity_score"] * 0.50, 0, 1)
		items["serving_priority"] = np.clip(items["retrieval_score"] * 0.50 + items["ranking_score"] * 0.50, 0, 1)
		self.content_embedding_matrix = self._normalized_embedding(len(items), self.config.embedding_dim)
		self.genre_embedding_matrix = self._normalized_embedding(len(items), 32)
		self.creator_embedding_matrix = self._normalized_embedding(len(items), 32)
		return items

	def _normalized_embedding(self, rows: int, dimensions: int) -> np.ndarray:
		vectors = self.rng.normal(0, 1, (rows, dimensions))
		return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

	def enrich_final_features(self, items: pd.DataFrame) -> pd.DataFrame:
		items = items.copy()
		new_item = items["content_age_days"].to_numpy() <= 14
		items["cold_start_item_flag"] = new_item
		items["cold_start_type"] = "none"
		items.loc[new_item, "cold_start_type"] = self.rng.choice(["new_movie", "new_series", "new_catalog_entry"], new_item.sum(), p=[0.40, 0.40, 0.20])
		return items

	def validate_items(self, items: pd.DataFrame) -> None:
		required = {"item_id", "genre", "language", "retrieval_score", "ranking_score"}
		missing = required.difference(items.columns)
		if missing:
			raise ValueError(f"missing item columns: {sorted(missing)}")
		if items.empty or items["item_id"].duplicated().any():
			raise ValueError("invalid item identifiers")
		for column in ["quality_score", "popularity_score", "trend_score", "retrieval_score", "ranking_score"]:
			if not items[column].between(0, 1).all():
				raise ValueError(f"invalid range in {column}")

	def build_two_tower_features(self, items: pd.DataFrame) -> pd.DataFrame:
		return items[["item_id", "genre", "language", "content_type", "quality_score", "popularity_score", "trend_score", "embedding_cluster"]].copy()

	def build_lightgcn_features(self, items: pd.DataFrame) -> pd.DataFrame:
		return items[["item_id", "genre", "creator", "embedding_cluster"]].copy()

	def build_item_feature_store(self, items: pd.DataFrame) -> pd.DataFrame:
		features = items.copy()
		features["event_timestamp"] = pd.Timestamp.utcnow()
		return features

	def export_item_embeddings(self) -> pd.DataFrame:
		if self.content_embedding_matrix is None:
			raise RuntimeError("generate items before exporting embeddings")
		embeddings = pd.DataFrame(self.content_embedding_matrix)
		embeddings.insert(0, "item_id", np.arange(1, len(embeddings) + 1))
		return embeddings

	def build_serving_features(self, items: pd.DataFrame) -> pd.DataFrame:
		return items[["item_id", "retrieval_score", "ranking_score", "serving_priority", "availability_status", "catalog_status", "trend_score"]].copy()

	def generate(self) -> pd.DataFrame:
		items = self.generate_base_items()
		items = self.enrich_quality_features(items)
		items = self.enrich_lifecycle_features(items)
		items = self.enrich_recommendation_features(items)
		items = self.enrich_final_features(items)
		self.validate_items(items)
		return items
