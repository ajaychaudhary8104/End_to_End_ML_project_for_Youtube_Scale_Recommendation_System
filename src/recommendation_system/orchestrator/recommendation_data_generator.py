from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..affinity import AffinityEngine
from ..bias import ContextBiasEngine, PositionBiasEngine
from ..datasets import TrainingDatasetBuilder
from ..feature_store import FeatureStoreBuilder
from ..features import ContextFeatureBuilder, ItemFeatureBuilder, UserFeatureBuilder
from ..foundation import FoundationBuilder, GeneratorConfig
from ..generators import (
	ColdStartEngine,
	ContextGenerator,
	EventStreamGenerator,
	InteractionGenerator,
	ItemGenerator,
	ReviewGenerator,
	SessionTimelineGenerator,
	UserGenerator,
	WatchHistoryGenerator,
)
from ..generators.cold_start_engine import ColdStartEngine
from ..graph import GraphRecommendationBuilder
from ..ranking.ranking_dataset import RankingGenerator
from ..reinforcement_learning import RLDataSetBuilder
from ..reranking.reranking_dataset import ReRankingGenerator
from ..retrieval.retrieval_builder import RetrievalGenerator
from ..search import SearchGenerator
from ..sequential import SequentialRecommendationGenerator
from ..bandits.contextual_bandit_dataset import BanditGenerator


class RecommendationDataGenerator:
	"""End-to-end modular synthetic recommendation data orchestrator."""

	def __init__(self, config: GeneratorConfig | None = None):
		self.config = config or GeneratorConfig(n_users=100, n_items=100, embedding_dim=32)
		self.foundation_builder = FoundationBuilder(self.config)
		self.foundation = self.foundation_builder.foundation
		self.assets: dict[str, object] = {}

	def generate_foundation(self):
		self.assets.update(self.foundation_builder.generate())
		return self.assets

	def generate_users(self):
		self._require_foundation()
		self.assets["users"] = UserGenerator(self.foundation, self.assets["personas"], self.assets["archetypes"], self.assets["user_embeddings"]).generate()
		return self.assets["users"]

	def generate_items(self):
		self._require_foundation()
		generator = ItemGenerator(self.foundation, self.assets["item_embeddings"])
		self.assets["items"] = generator.generate()
		self.assets["item_generator"] = generator
		return self.assets["items"]

	def generate_contexts(self, n_contexts: int | None = None):
		self._require("users")
		self.assets["contexts"] = ContextGenerator(self.foundation, self.assets["users"]).generate(n_contexts)
		return self.assets["contexts"]

	def generate_affinities(self, candidates_per_user: int = 50):
		self._require("users", "items", "contexts")
		generator = self.assets["item_generator"]
		self.assets["affinities"] = AffinityEngine(self.foundation, self.assets["users"], self.assets["items"], self.assets["user_embeddings"], generator.export_item_embeddings()).generate_affinity_dataset(self.assets["contexts"], candidates_per_user)
		return self.assets["affinities"]

	def generate_sessions(self):
		self._require("users", "contexts")
		self.assets["sessions"] = SessionTimelineGenerator(self.foundation, self.assets["users"], self.assets["contexts"]).generate()
		return self.assets["sessions"]

	def generate_interactions(self):
		self._require("sessions", "items", "affinities")
		self.assets["interactions"] = InteractionGenerator(self.foundation, self.assets["sessions"], self.assets["affinities"], self.assets["items"]).generate()
		return self.assets["interactions"]

	def generate_downstream_events(self):
		self._require("users", "items", "interactions")
		self.assets["watch_history"] = WatchHistoryGenerator(self.foundation, self.assets["users"], self.assets["items"], self.assets["interactions"]).generate()
		self.assets["event_stream"] = EventStreamGenerator(self.foundation, self.assets["users"], self.assets["items"], self.assets["interactions"]).generate()
		self.assets["reviews"] = ReviewGenerator(self.foundation, self.assets["users"], self.assets["items"], self.assets["interactions"]).generate()
		return {key: self.assets[key] for key in ["watch_history", "event_stream", "reviews"]}

	def generate_model_datasets(self):
		self._require("users", "items", "contexts", "affinities", "interactions")
		affinities = self.assets["affinities"]
		users, items = self.assets["users"], self.assets["items"]
		self.assets["position_bias"] = PositionBiasEngine(self.foundation, self.assets["contexts"]).generate()
		self.assets["context_bias"] = ContextBiasEngine(self.foundation, self.assets["contexts"], affinities).generate()
		self.assets["retrieval"] = RetrievalGenerator(self.foundation, users, items, affinities).generate()
		self.assets["ranking"] = RankingGenerator(self.foundation, users, items, affinities).generate()
		self.assets["reranking"] = ReRankingGenerator(self.foundation, self.assets["ranking"], items).generate()
		self.assets["search"] = SearchGenerator(self.foundation, users, items, affinities).generate()
		self.assets["bandit"] = BanditGenerator(self.foundation, users, items, affinities).generate()
		self.assets["rl"] = RLDataSetBuilder(self.foundation, users, items, self.assets["interactions"], affinities).generate()
		self.assets["cold_start"] = ColdStartEngine(self.foundation, users, items).generate()
		self.assets["sequential"] = SequentialRecommendationGenerator(self.foundation, users, items, self.assets["interactions"]).generate()
		self.assets["graph"] = GraphRecommendationBuilder(self.foundation, users, items, self.assets["interactions"]).generate()
		return self.assets

	def generate_feature_layers(self):
		self._require("users", "items", "contexts", "affinities")
		self.assets["user_features"] = UserFeatureBuilder().generate(self.assets["users"])
		self.assets["item_features"] = ItemFeatureBuilder().generate(self.assets["items"])
		self.assets["context_features"] = ContextFeatureBuilder().generate(self.assets["contexts"])
		self.assets["feature_store"] = FeatureStoreBuilder(self.foundation, self.assets["user_features"], self.assets["item_features"], self.assets["context_features"], self.assets["affinities"]).generate()
		self.assets["training_datasets"] = TrainingDatasetBuilder().generate(self.assets["affinities"])
		return self.assets["feature_store"]

	def generate_all(self):
		self.generate_foundation()
		self.generate_users()
		self.generate_items()
		self.generate_contexts()
		self.generate_affinities()
		self.generate_sessions()
		self.generate_interactions()
		self.generate_downstream_events()
		self.generate_model_datasets()
		self.generate_feature_layers()
		return self.assets

	def export(self, output_dir: str | Path | None = None):
		destination = Path(output_dir or self.config.output_dir)
		destination.mkdir(parents=True, exist_ok=True)
		exported = {}
		for name, value in self.assets.items():
			if isinstance(value, pd.DataFrame):
				path = destination / f"{name}.parquet"
				value.to_parquet(path, index=False)
				exported[name] = path
			elif isinstance(value, dict):
				for child_name, child_value in value.items():
					if isinstance(child_value, pd.DataFrame):
						path = destination / f"{child_name}.parquet"
						child_value.to_parquet(path, index=False)
						exported[child_name] = path
		return exported

	def _require_foundation(self):
		if not self.assets:
			self.generate_foundation()

	def _require(self, *names):
		missing = [name for name in names if name not in self.assets]
		if missing:
			raise ValueError(f"generate prerequisite assets first: {missing}")


def main():
	generator = RecommendationDataGenerator()
	generator.generate_all()
	generator.export()
	return generator
