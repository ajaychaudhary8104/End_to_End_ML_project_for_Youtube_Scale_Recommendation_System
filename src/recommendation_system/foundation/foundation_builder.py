from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random
import uuid

import numpy as np
import pandas as pd


@dataclass(slots=True)
class GeneratorConfig:
	n_users: int = 100_000
	n_items: int = 10_000
	embedding_dim: int = 128
	start_date: str = "2023-01-01"
	end_date: str = "2025-01-01"
	random_state: int = 42
	output_dir: str = "artifacts"
	min_personas: int = 8
	max_personas: int = 15
	min_archetypes: int = 5
	max_archetypes: int = 10
	normalize_embeddings: bool = True

	def validate(self) -> None:
		if self.n_users <= 0:
			raise ValueError("n_users must be > 0")
		if self.n_items <= 0:
			raise ValueError("n_items must be > 0")
		if self.embedding_dim <= 0:
			raise ValueError("embedding_dim must be > 0")
		if self.min_personas <= 0 or self.max_personas < self.min_personas:
			raise ValueError("persona bounds are invalid")
		if self.min_archetypes <= 0 or self.max_archetypes < self.min_archetypes:
			raise ValueError("archetype bounds are invalid")
		if self.max_personas > 15:
			raise ValueError("max_personas cannot exceed the available personas")
		if self.max_archetypes > 10:
			raise ValueError("max_archetypes cannot exceed the available archetypes")

		start = pd.Timestamp(self.start_date)
		end = pd.Timestamp(self.end_date)
		if start > end:
			raise ValueError("start_date must be before end_date")


class FoundationLayer:
	def __init__(self, config: GeneratorConfig):
		config.validate()
		self.config = config
		self.rng = np.random.default_rng(config.random_state)
		random.seed(config.random_state)
		np.random.seed(config.random_state)
		self.output_dir = Path(config.output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.run_id = str(uuid.uuid4())

	@property
	def date_range(self) -> pd.DatetimeIndex:
		return pd.date_range(
			start=self.config.start_date,
			end=self.config.end_date,
			freq="D",
		)

	def random_dates(self, size: int) -> np.ndarray:
		return self.rng.choice(self.date_range, size=size, replace=True)

	@staticmethod
	def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
		norms = np.linalg.norm(vectors, axis=1, keepdims=True)
		return vectors / np.maximum(norms, 1e-12)

	def create_metadata(self) -> dict[str, Any]:
		return {
			"run_id": self.run_id,
			"n_users": self.config.n_users,
			"n_items": self.config.n_items,
			"embedding_dim": self.config.embedding_dim,
			"random_state": self.config.random_state,
			"start_date": self.config.start_date,
			"end_date": self.config.end_date,
		}


class FoundationBuilder:
	def __init__(self, config: GeneratorConfig):
		from .archetype_engine import ArchetypeEngine
		from .latent_factor_engine import LatentFactorEngine
		from .persona_engine import PersonaEngine

		self.foundation = FoundationLayer(config)
		self.persona_engine = PersonaEngine(self.foundation)
		self.archetype_engine = ArchetypeEngine(self.foundation)
		self.latent_factor_engine = LatentFactorEngine(self.foundation)

	def generate(self) -> dict[str, Any]:
		personas = self.persona_engine.generate()
		archetypes = self.archetype_engine.generate()
		user_embeddings, item_embeddings = (
			self.latent_factor_engine.build_embedding_tables()
		)
		return {
			"metadata": self.foundation.create_metadata(),
			"personas": personas,
			"archetypes": archetypes,
			"user_embeddings": user_embeddings,
			"item_embeddings": item_embeddings,
		}


Phase1FoundationLayer = FoundationBuilder
