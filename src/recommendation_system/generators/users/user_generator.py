from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ...foundation import FoundationLayer


class UserGenerator:
	"""Generate demographic, behavioral, preference, and user-quality data."""

	GENRES = [
		"Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror",
		"Mystery", "Romance", "SciFi", "Thriller", "Documentary",
		"Animation", "Sports", "News", "Music",
	]
	DEVICES = ["Mobile", "Tablet", "Desktop", "SmartTV", "GamingConsole"]

	def __init__(
		self,
		foundation: FoundationLayer,
		personas: pd.DataFrame,
		archetypes: pd.DataFrame,
		user_embeddings: pd.DataFrame,
	):
		self.foundation = foundation
		self.config = foundation.config
		self.rng = foundation.rng
		self.personas = personas
		self.archetypes = archetypes
		self.user_embeddings = user_embeddings

		self._validate_reference_tables()

	def _validate_reference_tables(self) -> None:
		for table, name, key in (
			(self.personas, "personas", "persona_id"),
			(self.archetypes, "archetypes", "archetype_id"),
			(self.user_embeddings, "user_embeddings", "user_id"),
		):
			if key not in table.columns:
				raise ValueError(f"{name} must contain {key}")

	def _lookup(self, table: pd.DataFrame) -> pd.DataFrame:
		return table.set_index(table.columns[0])

	def _generate_age(self) -> np.ndarray:
		buckets = [(13, 17), (18, 24), (25, 34), (35, 44), (45, 54), (55, 70)]
		selected = self.rng.choice(
			len(buckets),
			size=self.config.n_users,
			p=[0.08, 0.20, 0.28, 0.20, 0.15, 0.09],
		)
		ages = np.zeros(self.config.n_users, dtype=np.int16)
		for index, (low, high) in enumerate(buckets):
			mask = selected == index
			ages[mask] = self.rng.integers(low, high + 1, mask.sum())
		return ages

	@staticmethod
	def _generate_age_group(ages: np.ndarray) -> pd.Series:
		return pd.cut(
			ages,
			bins=[0, 18, 25, 35, 45, 55, 100],
			labels=["teen", "young_adult", "adult", "mid_age", "mature", "senior"],
			include_lowest=True,
		).astype(str)

	def _generate_geography(self) -> pd.DataFrame:
		countries = np.array(
			["USA", "India", "UK", "Canada", "Germany", "Brazil", "Japan", "Australia"]
		)
		sampled = self.rng.choice(
			countries,
			size=self.config.n_users,
			p=[0.30, 0.25, 0.10, 0.08, 0.08, 0.07, 0.07, 0.05],
		)
		languages = {
			"USA": "English", "India": "Hindi", "UK": "English",
			"Canada": "English", "Germany": "German", "Brazil": "Portuguese",
			"Japan": "Japanese", "Australia": "English",
		}
		return pd.DataFrame(
			{
				"country": sampled,
				"language": [languages[country] for country in sampled],
				"region": self.rng.choice(
					["Urban", "Suburban", "Rural"],
					size=self.config.n_users,
					p=[0.60, 0.30, 0.10],
				),
			}
		)

	def _generate_registration_dates(self) -> pd.Series:
		start = pd.Timestamp(self.config.start_date)
		end = pd.Timestamp(self.config.end_date)
		total_days = max((end - start).days, 1)
		offsets = self.rng.integers(0, total_days, self.config.n_users)
		return pd.Series(start + pd.to_timedelta(offsets, unit="D"))

	def _generate_households(self) -> pd.DataFrame:
		sizes = self.rng.choice(
			[1, 2, 3, 4, 5],
			size=self.config.n_users,
			p=[0.30, 0.28, 0.20, 0.15, 0.07],
		)
		return pd.DataFrame(
			{
				"household_id": [f"H{index:08d}" for index in range(1, len(sizes) + 1)],
				"household_size": sizes,
			}
		)

	def _generate_subscription_plan(self) -> pd.DataFrame:
		plans = self.rng.choice(
			["free", "basic", "standard", "premium", "family"],
			size=self.config.n_users,
			p=[0.15, 0.25, 0.30, 0.20, 0.10],
		)
		prices = {"free": 0.0, "basic": 7.99, "standard": 12.99, "premium": 19.99, "family": 24.99}
		return pd.DataFrame(
			{
				"subscription_plan": plans,
				"monthly_price": [prices[plan] for plan in plans],
				"is_premium": np.isin(plans, ["premium", "family"]),
			}
		)

	def generate_base_users(self) -> pd.DataFrame:
		ages = self._generate_age()
		users = pd.DataFrame(
			{
				"user_id": np.arange(1, self.config.n_users + 1),
				"persona_id": self.rng.choice(self.personas["persona_id"].values, self.config.n_users),
				"archetype_id": self.rng.choice(self.archetypes["archetype_id"].values, self.config.n_users),
				"age": ages,
				"age_group": self._generate_age_group(ages),
				"gender": self.rng.choice(["male", "female", "other"], self.config.n_users, p=[0.49, 0.49, 0.02]),
			}
		)
		registration = self._generate_registration_dates()
		users = pd.concat(
			[users, self._generate_geography(), self._generate_households(), self._generate_subscription_plan()],
			axis=1,
		)
		users["registration_date"] = registration
		users["tenure_days"] = (pd.Timestamp(self.config.end_date) - registration).dt.days.astype(np.int32)
		users["embedding_id"] = users["user_id"]
		return users

	def _persona_lookup(self) -> pd.DataFrame:
		return self.personas.set_index("persona_id")

	def _archetype_lookup(self) -> pd.DataFrame:
		return self.archetypes.set_index("archetype_id")

	def enrich_behavioral_features(self, users: pd.DataFrame) -> pd.DataFrame:
		persona = self._persona_lookup()
		archetype = self._archetype_lookup()
		engagement = np.clip(
			users["persona_id"].map(persona["engagement_score"]).to_numpy()
			+ np.clip(users["tenure_days"].to_numpy() / 1000, 0, 1) * 0.15
			+ users["is_premium"].astype(int).to_numpy() * 0.08
			+ self.rng.normal(0, 0.05, len(users)),
			0.01,
			1.0,
		)
		churn = np.clip(
			users["archetype_id"].map(archetype["churn_probability"]).to_numpy()
			+ (1 - engagement) * 0.60
			+ (1 - np.clip(users["tenure_days"].to_numpy() / 1000, 0, 1)) * 0.15,
			0.01,
			1.0,
		)
		price_max = max(float(users["monthly_price"].max()), 1.0)
		ltv = np.clip(
			0.40 * engagement
			+ 0.30 * np.clip(users["tenure_days"].to_numpy() / 1000, 0, 1)
			+ 0.30 * users["monthly_price"].to_numpy() / price_max,
			0,
			1,
		)
		avg_session = np.clip(
			users["persona_id"].map(persona["avg_session_length"]).to_numpy()
			+ self.rng.normal(0, 10, len(users)),
			5,
			300,
		)
		daily_sessions = np.maximum(1, self.rng.poisson(0.5 + engagement * 5))
		weekend_ratio = np.clip(self.rng.beta(3, 2, len(users)), 0.10, 0.95)
		night_ratio = np.where(
			users["age"].to_numpy() < 30,
			self.rng.normal(0.60, 0.15, len(users)),
			self.rng.normal(0.35, 0.10, len(users)),
		)
		night_ratio = np.clip(night_ratio, 0.01, 0.95)
		exploration = np.clip(
			users["persona_id"].map(persona["exploration_rate"]).to_numpy()
			+ self.rng.normal(0, 0.05, len(users)),
			0.01,
			0.95,
		)
		loyalty = np.clip(engagement * (1 - churn), 0, 1)
		users = users.copy()
		users["engagement_score"] = engagement
		users["churn_risk"] = churn
		users["lifetime_value_score"] = ltv
		users["avg_session_length"] = avg_session
		users["avg_daily_sessions"] = daily_sessions
		users["weekend_activity_ratio"] = weekend_ratio
		users["night_activity_ratio"] = night_ratio
		users["exploration_rate"] = exploration
		users["loyalty_score"] = loyalty
		users["power_user_flag"] = (engagement > 0.80) & (daily_sessions >= 4)
		users["cold_start_flag"] = users["tenure_days"] < 30
		users["marketing_segment"] = np.select(
			[ltv > 0.80, churn > 0.70, (engagement > 0.85) & (ltv > 0.70)],
			["vip", "retention", "power"],
			default="standard",
		)
		return users

	def _generate_device_ownership(self, users: pd.DataFrame) -> pd.DataFrame:
		primary = []
		secondary = []
		for age in users["age"]:
			if age < 25:
				device = self.rng.choice(["Mobile", "Desktop"], p=[0.75, 0.25])
			elif age < 45:
				device = self.rng.choice(["Mobile", "SmartTV", "Desktop"], p=[0.45, 0.35, 0.20])
			else:
				device = self.rng.choice(["SmartTV", "Tablet", "Mobile"], p=[0.55, 0.15, 0.30])
			primary.append(device)
			secondary.append(self.rng.choice([item for item in self.DEVICES if item != device]))
		return pd.DataFrame({"primary_device": primary, "secondary_device": secondary})

	def enrich_preference_features(self, users: pd.DataFrame) -> pd.DataFrame:
		users = users.copy()
		language = {"USA": "English", "India": "Hindi", "UK": "English", "Canada": "English", "Germany": "German", "Brazil": "Portuguese", "Japan": "Japanese", "Australia": "English"}
		users["preferred_language"] = users["country"].map(language)
		users = pd.concat([users, self._generate_device_ownership(users)], axis=1)
		users["preferred_genres"] = [
			list(self.rng.choice(self.GENRES, size=self.rng.integers(2, 6), replace=False))
			for _ in range(len(users))
		]
		users["content_maturity_preference"] = [
			"Kids" if age < 18 else (self.rng.choice(["Teen", "Adult"], p=[0.40, 0.60]) if age < 25 else "Adult")
			for age in users["age"]
		]
		users["popularity_preference"] = np.clip(0.4 + users["engagement_score"] * 0.5 + self.rng.normal(0, 0.1, len(users)), 0, 1)
		users["novelty_preference"] = np.clip(users["exploration_rate"] + self.rng.normal(0, 0.05, len(users)), 0, 1)
		users["user_embedding_cluster"] = self.rng.integers(0, 20, len(users))
		users["taste_profile"] = [
			"Family" if age < 18 else (
				self.rng.choice(["PremiumViewer", "Explorer", "BingeWatcher"])
				if premium else (
					"Explorer" if exploration > 0.50 else (
						"BingeWatcher" if engagement > 0.80 else self.rng.choice(["Mainstream", "TrendFollower", "Blockbuster", "Niche"])
					)
				)
			)
			for age, premium, exploration, engagement in zip(
				users["age"], users["is_premium"], users["exploration_rate"], users["engagement_score"]
			)
		]
		vectors = self.rng.gamma(2.0, 1.0, size=(len(users), len(self.GENRES)))
		self.genre_affinity_matrix = vectors / vectors.sum(axis=1, keepdims=True)
		return users

	def enrich_final_features(self, users: pd.DataFrame) -> pd.DataFrame:
		users = users.copy()
		users["household_role"] = [
			"Teen" if age < 18 else ("Primary" if size == 1 or self.rng.random() < 0.25 else "Adult")
			for age, size in zip(users["age"], users["household_size"])
		]
		users["is_family_account"] = (users["subscription_plan"] == "family") | (users["household_size"] >= 3)
		users["is_kids_profile"] = users["age"] < 13
		recent = users["tenure_days"] <= 30
		users["cold_start_type"] = "none"
		users.loc[recent, "cold_start_type"] = self.rng.choice(["new_user", "sparse_user"], recent.sum(), p=[0.7, 0.3])
		users["user_quality_score"] = np.clip(
			users["engagement_score"] * 0.35 + users["loyalty_score"] * 0.35 + users["lifetime_value_score"] * 0.30 + self.rng.normal(0, 0.03, len(users)),
			0,
			1,
		)
		users["user_segment"] = np.select(
			[users["lifetime_value_score"] > 0.80, users["churn_risk"] > 0.70, users["engagement_score"] > 0.80, users["cold_start_flag"]],
			["high_value", "at_risk", "engaged", "new_user"],
			default="standard",
		)
		return users

	def attach_user_embeddings(self, users: pd.DataFrame) -> pd.DataFrame:
		return users.merge(
			self.user_embeddings,
			left_on="embedding_id",
			right_on="user_id",
			how="left",
			suffixes=("", "_embedding"),
		)

	def build_user_feature_store(self, users: pd.DataFrame) -> pd.DataFrame:
		features = users.copy()
		features["event_timestamp"] = pd.Timestamp.utcnow()
		return features

	def validate(self, users: pd.DataFrame) -> None:
		required = {"user_id", "age", "subscription_plan"}
		missing = required.difference(users.columns)
		if missing:
			raise ValueError(f"missing required user columns: {sorted(missing)}")
		if users.empty or users["user_id"].duplicated().any() or users["age"].isna().any() or users["subscription_plan"].isna().any():
			raise ValueError("invalid base users")

	def validate_final(self, users: pd.DataFrame) -> None:
		if users.isna().sum().sum() > 0:
			raise ValueError("Missing values detected.")
		for column in ["engagement_score", "churn_risk", "user_quality_score"]:
			if not users[column].between(0, 1).all():
				raise ValueError(f"Invalid {column}.")

	def generate(self) -> pd.DataFrame:
		users = self.generate_base_users()
		users = self.enrich_behavioral_features(users)
		users = self.enrich_preference_features(users)
		users = self.enrich_final_features(users)
		self.validate(users)
		self.validate_final(users)
		return users
