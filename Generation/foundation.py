from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import random
import uuid

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class GeneratorConfig:
    """
    Global configuration for the recommendation data generator.

    Parameters
    ----------
    n_users : int
        Number of users to generate.

    n_items : int
        Number of items to generate.

    embedding_dim : int
        Latent embedding dimension.

    start_date : str
        Dataset start date.

    end_date : str
        Dataset end date.

    random_state : int
        Global seed.

    output_dir : str
        Export directory.

    """

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
        """
        Validate configuration.
        """

        if self.n_users <= 0:
            raise ValueError("n_users must be > 0")

        if self.n_items <= 0:
            raise ValueError("n_items must be > 0")

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")

        pd.Timestamp(self.start_date)
        pd.Timestamp(self.end_date)


# ============================================================
# FOUNDATION LAYER
# ============================================================

class FoundationLayer:
    """
    Foundation layer shared by all generators.

    Responsibilities
    ----------------
    - Configuration validation
    - Seed management
    - Random number generation
    - Date utilities
    - Export utilities
    - Global metadata creation
    """

    def __init__(self, config: GeneratorConfig):

        config.validate()

        self.config = config

        self.rng = np.random.default_rng(
            config.random_state
        )

        random.seed(config.random_state)

        np.random.seed(config.random_state)

        self.output_dir = Path(config.output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.run_id = str(uuid.uuid4())

    @property
    def date_range(self) -> pd.DatetimeIndex:
        """
        Full simulation date range.
        """

        return pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq="D"
        )

    def random_dates(
        self,
        size: int
    ) -> np.ndarray:
        """
        Sample random dates.
        """

        return self.rng.choice(
            self.date_range,
            size=size,
            replace=True
        )

    def normalize_vectors(
        self,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        L2 normalize vectors.
        """

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        norms = np.maximum(norms, 1e-12)

        return vectors / norms

    def create_metadata(self) -> Dict[str, Any]:
        """
        Build generation metadata.
        """

        return {
            "run_id": self.run_id,
            "n_users": self.config.n_users,
            "n_items": self.config.n_items,
            "embedding_dim": self.config.embedding_dim,
            "random_state": self.config.random_state,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date
        }


# ============================================================
# PERSONA ENGINE
# ============================================================

class PersonaEngine:
    """
    Generates behavioral personas.

    Personas represent user behavior
    before actual user generation.

    Examples
    --------
    Casual Viewer
    Binge Watcher
    Sports Fan
    Documentary Lover
    Family User
    Kids User
    Anime Fan
    Premium Power User
    """

    DEFAULT_PERSONAS = [
        "Casual Viewer",
        "Binge Watcher",
        "Weekend User",
        "Sports Fan",
        "Movie Enthusiast",
        "Documentary Lover",
        "Anime Fan",
        "Family User",
        "Kids User",
        "Premium Power User",
        "Trend Chaser",
        "News Consumer",
        "Music Addict",
        "Mobile User",
        "Creator Follower"
    ]

    def __init__(
        self,
        foundation: FoundationLayer
    ):

        self.foundation = foundation
        self.rng = foundation.rng

    def generate(self) -> pd.DataFrame:
        """
        Generate persona definitions.
        """

        n_personas = self.rng.integers(
            self.foundation.config.min_personas,
            self.foundation.config.max_personas + 1
        )

        selected = self.DEFAULT_PERSONAS[:n_personas]

        persona_df = pd.DataFrame({
            "persona_id": np.arange(
                1,
                len(selected) + 1
            ),
            "persona_name": selected,
            "avg_session_length": self.rng.uniform(
                10,
                180,
                len(selected)
            ),
            "daily_active_probability": self.rng.uniform(
                0.05,
                0.95,
                len(selected)
            ),
            "exploration_rate": self.rng.uniform(
                0.01,
                0.50,
                len(selected)
            ),
            "engagement_score": self.rng.uniform(
                0.20,
                1.00,
                len(selected)
            )
        })

        return persona_df


# ============================================================
# LATENT FACTOR ENGINE
# ============================================================

class LatentFactorEngine:
    """
    Generates latent embeddings.

    These embeddings become the hidden
    preference space used later by:

    - Affinity Engine
    - Retrieval Models
    - Ranking Models
    - Graph Models
    - Sequential Models
    """

    def __init__(
        self,
        foundation: FoundationLayer
    ):

        self.foundation = foundation
        self.rng = foundation.rng

    def generate_user_embeddings(
        self
    ) -> np.ndarray:
        """
        Generate user latent vectors.
        """

        embeddings = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(
                self.foundation.config.n_users,
                self.foundation.config.embedding_dim
            )
        )

        if self.foundation.config.normalize_embeddings:
            embeddings = (
                self.foundation
                .normalize_vectors(
                    embeddings
                )
            )

        return embeddings

    def generate_item_embeddings(
        self
    ) -> np.ndarray:
        """
        Generate item latent vectors.
        """

        embeddings = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(
                self.foundation.config.n_items,
                self.foundation.config.embedding_dim
            )
        )

        if self.foundation.config.normalize_embeddings:
            embeddings = (
                self.foundation
                .normalize_vectors(
                    embeddings
                )
            )

        return embeddings

    def build_embedding_tables(
        self
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create embedding DataFrames.
        """

        user_emb = self.generate_user_embeddings()
        item_emb = self.generate_item_embeddings()

        user_df = pd.DataFrame(
            user_emb
        )

        user_df.insert(
            0,
            "user_id",
            np.arange(
                1,
                len(user_df) + 1
            )
        )

        item_df = pd.DataFrame(
            item_emb
        )

        item_df.insert(
            0,
            "item_id",
            np.arange(
                1,
                len(item_df) + 1
            )
        )

        return user_df, item_df


# ============================================================
# ARCHETYPE ENGINE
# ============================================================

class ArchetypeEngine:
    """
    Generates higher-level behavioral archetypes.

    Archetypes are broader than personas.

    Example
    -------
    Explorer
    Loyalist
    Passive Consumer
    Heavy Consumer
    Trend Seeker
    """

    DEFAULT_ARCHETYPES = [
        "Explorer",
        "Loyalist",
        "Passive Consumer",
        "Heavy Consumer",
        "Trend Seeker",
        "Niche Consumer",
        "Social Influenced",
        "Seasonal User",
        "High Value User",
        "Churn Risk User"
    ]

    def __init__(
        self,
        foundation: FoundationLayer
    ):

        self.foundation = foundation
        self.rng = foundation.rng

    def generate(self) -> pd.DataFrame:
        """
        Generate archetype definitions.
        """

        n_arch = self.rng.integers(
            self.foundation.config.min_archetypes,
            self.foundation.config.max_archetypes + 1
        )

        selected = self.DEFAULT_ARCHETYPES[:n_arch]

        archetypes = pd.DataFrame({
            "archetype_id": np.arange(
                1,
                len(selected) + 1
            ),
            "archetype_name": selected,
            "retention_score": self.rng.uniform(
                0.20,
                1.00,
                len(selected)
            ),
            "churn_probability": self.rng.uniform(
                0.01,
                0.60,
                len(selected)
            ),
            "average_ltv": self.rng.uniform(
                100,
                5000,
                len(selected)
            ),
            "content_diversity": self.rng.uniform(
                0.10,
                1.00,
                len(selected)
            )
        })

        return archetypes


# ============================================================
# PHASE 1 ORCHESTRATOR
# ============================================================

class Phase1FoundationLayer:
    """
    Complete Phase-1 execution layer.

    Executes:
        FoundationLayer
        PersonaEngine
        LatentFactorEngine
        ArchetypeEngine

    Returns all foundational assets
    required by downstream phases.
    """

    def __init__(
        self,
        config: GeneratorConfig
    ):

        self.foundation = FoundationLayer(
            config
        )

        self.persona_engine = PersonaEngine(
            self.foundation
        )

        self.latent_engine = LatentFactorEngine(
            self.foundation
        )

        self.archetype_engine = ArchetypeEngine(
            self.foundation
        )

    def generate(self) -> Dict[str, Any]:
        """
        Execute complete Phase 1.
        """

        personas = (
            self.persona_engine.generate()
        )

        archetypes = (
            self.archetype_engine.generate()
        )

        user_embeddings, item_embeddings = (
            self.latent_engine
            .build_embedding_tables()
        )

        metadata = (
            self.foundation
            .create_metadata()
        )

        return {
            "metadata": metadata,
            "personas": personas,
            "archetypes": archetypes,
            "user_embeddings": user_embeddings,
            "item_embeddings": item_embeddings
        }


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    config = GeneratorConfig(
        n_users=100_000,
        n_items=10_000,
        embedding_dim=128,
        random_state=42
    )

    phase1 = Phase1FoundationLayer(
        config
    )

    assets = phase1.generate()

    print(
        assets["personas"].head()
    )

    print(
        assets["archetypes"].head()
    )

    print(
        assets["user_embeddings"].shape
    )

    print(
        assets["item_embeddings"].shape
    )

    