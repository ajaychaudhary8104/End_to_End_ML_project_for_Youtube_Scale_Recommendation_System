from __future__ import annotations

import numpy as np
import pandas as pd

from ...foundation import FoundationLayer


class ContextGenerator:
	"""Generate temporal, device, business, and recommendation context."""

	DEVICE_TYPES = ["Mobile", "Desktop", "Tablet", "SmartTV", "Console"]
	NETWORK_TYPES = ["WiFi", "5G", "4G", "Ethernet"]
	OS_TYPES = ["Android", "iOS", "Windows", "MacOS", "Linux", "TVOS"]
	SURFACE_TYPES = ["homepage", "search", "details_page", "continue_watching", "trending", "recommended", "category_page"]
	TRAFFIC_SOURCES = ["organic", "push_notification", "email", "advertisement", "social", "direct"]
	SESSION_INTENTS = ["browse", "watch", "search", "continue", "explore"]

	def __init__(self, foundation: FoundationLayer, users: pd.DataFrame | None = None):
		self.foundation = foundation
		self.config = foundation.config
		self.rng = foundation.rng
		self.users = users
		self.context_embedding_matrix: np.ndarray | None = None

	def _generate_timestamps(self, count: int) -> pd.Series:
		start = pd.Timestamp(self.config.start_date)
		end = pd.Timestamp(self.config.end_date)
		total_seconds = max(int((end - start).total_seconds()), 1)
		offsets = self.rng.integers(0, total_seconds, size=count)
		return pd.Series(start + pd.to_timedelta(offsets, unit="s"))

	def generate_base_contexts(self, n_contexts: int) -> pd.DataFrame:
		timestamps = self._generate_timestamps(n_contexts)
		temporal = pd.DataFrame(
			{
				"timestamp": timestamps,
				"date": timestamps.dt.date,
				"hour_of_day": timestamps.dt.hour,
				"day_of_week": timestamps.dt.dayofweek,
				"week_of_year": timestamps.dt.isocalendar().week.astype(int),
				"month": timestamps.dt.month,
				"quarter": timestamps.dt.quarter,
			}
		)
		month = temporal["month"].to_numpy()
		temporal["season"] = np.select(
			[np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8])],
			["winter", "spring", "summer"],
			default="autumn",
		)
		contexts = pd.concat(
			[
				temporal,
				pd.DataFrame(
					{
						"device_type": self.rng.choice(self.DEVICE_TYPES, n_contexts, p=[0.40, 0.20, 0.10, 0.25, 0.05]),
						"network_type": self.rng.choice(self.NETWORK_TYPES, n_contexts, p=[0.55, 0.20, 0.15, 0.10]),
						"os_type": self.rng.choice(self.OS_TYPES, n_contexts),
					}
				),
				pd.DataFrame(
					{
						"traffic_source": self.rng.choice(self.TRAFFIC_SOURCES, n_contexts, p=[0.45, 0.10, 0.08, 0.07, 0.10, 0.20]),
						"campaign_id": self.rng.integers(1, 1000, n_contexts),
					}
				),
				pd.DataFrame(
					{
						"surface_type": self.rng.choice(self.SURFACE_TYPES, n_contexts),
						"recommendation_slot": self.rng.integers(1, 51, n_contexts),
						"page_position": self.rng.integers(1, 101, n_contexts),
					}
				),
				pd.DataFrame(
					{
						"session_intent": self.rng.choice(self.SESSION_INTENTS, n_contexts),
						"attention_level": self.rng.beta(3, 2, n_contexts),
						"mood_proxy": self.rng.choice(["relaxed", "focused", "casual", "exploratory"], n_contexts),
					}
				),
			],
			axis=1,
		)
		contexts.insert(0, "context_id", np.arange(1, n_contexts + 1))
		return contexts

	def enrich_business_context(self, contexts: pd.DataFrame) -> pd.DataFrame:
		contexts = contexts.copy()
		month = contexts["month"].to_numpy()
		holiday = self.rng.random(len(contexts)) < np.where(np.isin(month, [1, 11, 12]), 0.18, 0.03)
		special_event = self.rng.random(len(contexts)) < 0.05
		promotion = self.rng.random(len(contexts)) < 0.10
		countries = self.rng.choice(["USA", "India", "UK", "Germany", "Japan", "Canada", "Australia", "Brazil"], len(contexts), p=[0.30, 0.20, 0.10, 0.08, 0.07, 0.08, 0.07, 0.10])
		timezones = {
			"USA": "America/New_York", "India": "Asia/Kolkata", "UK": "Europe/London", "Germany": "Europe/Berlin",
			"Japan": "Asia/Tokyo", "Canada": "America/Toronto", "Australia": "Australia/Sydney", "Brazil": "America/Sao_Paulo",
		}
		demand = np.clip(np.sin(contexts["hour_of_day"].to_numpy() / 24 * np.pi) * 0.5 + 0.5 + self.rng.normal(0, 0.05, len(contexts)), 0, 1)
		network = contexts["network_type"].to_numpy()
		quality = np.zeros(len(contexts))
		for network_type, low, high in [("Ethernet", 0.90, 1.00), ("WiFi", 0.70, 1.00), ("5G", 0.75, 0.95), ("4G", 0.50, 0.85)]:
			mask = network == network_type
			quality[mask] = self.rng.uniform(low, high, mask.sum())
		event_types = np.full(len(contexts), "none", dtype=object)
		event_types[special_event] = self.rng.choice(["sports", "concert", "festival", "movie_launch", "award_show"], special_event.sum())
		campaigns = np.full(len(contexts), "none", dtype=object)
		campaigns[promotion] = self.rng.choice(["email", "push", "social", "paid_search", "affiliate"], promotion.sum())
		contexts["holiday_flag"] = holiday
		contexts["special_event_flag"] = special_event
		contexts["promotion_flag"] = promotion
		contexts["country_context"] = countries
		contexts["timezone_context"] = [timezones[country] for country in countries]
		contexts["real_time_demand_index"] = demand
		contexts["streaming_load_index"] = np.clip(demand + self.rng.normal(0, 0.05, len(contexts)), 0, 1)
		contexts["network_quality_score"] = quality
		contexts["weather_context"] = self.rng.choice(["sunny", "cloudy", "rainy", "stormy", "snowy"], len(contexts), p=[0.35, 0.25, 0.20, 0.05, 0.15])
		contexts["event_context"] = event_types
		contexts["ad_campaign_context"] = campaigns
		return contexts

	def enrich_recommendation_context(self, contexts: pd.DataFrame) -> pd.DataFrame:
		contexts = contexts.copy()
		surface = contexts["surface_type"].to_numpy()
		homepage = np.full(len(contexts), 0.3)
		homepage[surface == "homepage"] = self.rng.uniform(0.7, 1.0, (surface == "homepage").sum())
		search = np.full(len(contexts), 0.2)
		search[surface == "search"] = self.rng.uniform(0.7, 1.0, (surface == "search").sum())
		contexts["homepage_bias"] = homepage
		contexts["search_bias"] = search
		contexts["watch_intent_score"] = np.clip(contexts["attention_level"].to_numpy() * 0.6 + contexts["real_time_demand_index"].to_numpy() * 0.4 + self.rng.normal(0, 0.05, len(contexts)), 0, 1)
		contexts["purchase_intent_score"] = np.clip(self.rng.beta(2, 8, len(contexts)), 0, 1)
		contexts["session_depth"] = np.clip(np.round(self.rng.lognormal(2.0, 0.7, len(contexts)),), 1, 100).astype(np.int32)
		contexts["exploration_mode"] = self.rng.random(len(contexts)) < 0.30
		contexts["binge_mode"] = (contexts["session_intent"].to_numpy() == "continue") | (self.rng.random(len(contexts)) < 0.15)
		contexts["multi_screen_flag"] = self.rng.random(len(contexts)) < 0.12
		contexts["recommendation_competition"] = np.clip(contexts["page_position"].to_numpy() + self.rng.normal(0, 5, len(contexts)), 1, 100)
		contexts["content_supply_pressure"] = np.clip(self.rng.beta(5, 2, len(contexts)), 0, 1)
		contexts["ranking_pressure_score"] = np.clip(contexts["content_supply_pressure"].to_numpy() * 0.6 + contexts["recommendation_competition"].to_numpy() / 100 * 0.4, 0, 1)
		return contexts

	def enrich_final_context_features(self, contexts: pd.DataFrame) -> pd.DataFrame:
		contexts = contexts.copy()
		contexts["context_embedding_id"] = contexts["context_id"]
		contexts["context_cluster"] = self.rng.integers(0, 50, len(contexts))
		contexts["context_quality_score"] = np.clip(
			contexts["attention_level"].to_numpy() * 0.30
			+ contexts["network_quality_score"].to_numpy() * 0.25
			+ contexts["watch_intent_score"].to_numpy() * 0.25
			+ (1 - contexts["ranking_pressure_score"].to_numpy()) * 0.20
			+ self.rng.normal(0, 0.03, len(contexts)),
			0,
			1,
		)
		vectors = self.rng.normal(0, 1, (len(contexts), self.config.embedding_dim))
		self.context_embedding_matrix = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
		return contexts

	def build_context_feature_store(self, contexts: pd.DataFrame) -> pd.DataFrame:
		features = contexts.copy()
		features["event_timestamp"] = pd.Timestamp.utcnow()
		return features

	def build_serving_context_features(self, contexts: pd.DataFrame) -> pd.DataFrame:
		return contexts[["context_id", "watch_intent_score", "homepage_bias", "search_bias", "ranking_pressure_score", "context_quality_score", "real_time_demand_index", "network_quality_score"]].copy()

	def export_context_embeddings(self) -> pd.DataFrame:
		if self.context_embedding_matrix is None:
			raise RuntimeError("generate contexts before exporting embeddings")
		embeddings = pd.DataFrame(self.context_embedding_matrix)
		embeddings.insert(0, "context_id", np.arange(1, len(embeddings) + 1))
		return embeddings

	def validate_contexts(self, contexts: pd.DataFrame) -> None:
		required = {"context_id", "watch_intent_score", "ranking_pressure_score", "context_quality_score"}
		missing = required.difference(contexts.columns)
		if missing:
			raise ValueError(f"missing context columns: {sorted(missing)}")
		if contexts.empty or contexts["context_id"].duplicated().any():
			raise ValueError("invalid context identifiers")
		for column in ["watch_intent_score", "purchase_intent_score", "network_quality_score", "ranking_pressure_score", "context_quality_score"]:
			if not contexts[column].between(0, 1).all():
				raise ValueError(f"invalid values in {column}")

	def generate(self, n_contexts: int | None = None) -> pd.DataFrame:
		count = self.config.n_users if n_contexts is None else n_contexts
		if count <= 0:
			raise ValueError("n_contexts must be > 0")
		contexts = self.generate_base_contexts(count)
		contexts = self.enrich_business_context(contexts)
		contexts = self.enrich_recommendation_context(contexts)
		contexts = self.enrich_final_context_features(contexts)
		self.validate_contexts(contexts)
		return contexts
