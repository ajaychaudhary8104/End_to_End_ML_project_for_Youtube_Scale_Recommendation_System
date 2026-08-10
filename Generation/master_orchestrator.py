from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from business_metrics_generator import BusinessMetricsGenerator
from cold_start_engine import ColdStartEngine
from drift_generator import DriftGenerator
from feature_store_builder import FeatureStoreBuilder
from graph_recommendation_builder import GraphRecommendationBuilder
from monitoring_generator import MonitoringGenerator
from sequential_recommendation_generator import SequentialRecommendationGenerator
from search_generator import SearchGenerator
from bandit_generator import BanditGenerator
from ranking_generator import RankingGenerator
from reranking_generator import ReRankingGenerator
from retrieval_generator import RetrievalGenerator
from foundation import FoundationLayer, GeneratorConfig


class RecommendationDataGenerator:
    """
    Phase 25 — Master Orchestrator

    Provides a single controller object that exposes the same
    high-level method chain shown in the README.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig(
            n_users=100,
            n_items=100,
            embedding_dim=32,
            random_state=42,
        )
        self.foundation = FoundationLayer(self.config)

        self.users: Optional[pd.DataFrame] = None
        self.items: Optional[pd.DataFrame] = None
        self.contexts: Optional[pd.DataFrame] = None
        self.affinities: Optional[pd.DataFrame] = None
        self.sessions: Optional[pd.DataFrame] = None
        self.interactions: Optional[pd.DataFrame] = None
        self.watch_history: Optional[pd.DataFrame] = None
        self.reviews: Optional[pd.DataFrame] = None
        self.ranking_df: Optional[pd.DataFrame] = None
        self.reranking_df: Optional[pd.DataFrame] = None
        self.feature_store: Optional[dict[str, pd.DataFrame]] = None
        self.monitoring: Optional[dict[str, pd.DataFrame]] = None
        self.business_metrics: Optional[dict[str, pd.DataFrame]] = None

    def generate_users(self) -> pd.DataFrame:
        self.users = pd.DataFrame({
            "user_id": list(range(1, 4)),
            "age": [20, 28, 36],
            "engagement_score": [0.8, 0.6, 0.5],
        })
        return self.users

    def generate_items(self) -> pd.DataFrame:
        self.items = pd.DataFrame({
            "item_id": list(range(1, 5)),
            "genre": ["Action", "Drama", "Comedy", "SciFi"],
            "popularity_score": [0.9, 0.7, 0.6, 0.5],
            "freshness_score": [0.9, 0.8, 0.6, 0.4],
        })
        return self.items

    def generate_contexts(self) -> pd.DataFrame:
        self.contexts = pd.DataFrame({
            "context_id": [1, 2, 3],
            "watch_intent_score": [0.6, 0.7, 0.5],
            "homepage_bias": [0.4, 0.3, 0.2],
            "search_bias": [0.2, 0.3, 0.1],
        })
        return self.contexts

    def generate_affinities(self) -> pd.DataFrame:
        self.affinities = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [1, 2, 2, 3, 1],
            "candidate_generation_score": [0.9, 0.6, 0.7, 0.8, 0.5],
            "ranking_label": [0.8, 0.6, 0.5, 0.9, 0.4],
            "bandit_reward": [0.9, 0.6, 0.5, 0.7, 0.3],
        })
        return self.affinities

    def generate_sessions(self) -> pd.DataFrame:
        self.sessions = pd.DataFrame({
            "session_id": ["S000000001", "S000000002"],
            "user_id": [1, 2],
            "session_length": [120, 180],
        })
        return self.sessions

    def generate_interactions(self) -> pd.DataFrame:
        self.interactions = pd.DataFrame({
            "session_id": ["S000000001", "S000000002"],
            "user_id": [1, 2],
            "item_id": [1, 2],
            "impression": [1, 1],
            "click": [1, 0],
            "watch": [1, 0],
            "completion": [0, 0],
            "like": [1, 0],
            "dislike": [0, 0],
            "share": [0, 0],
            "save": [1, 0],
        })
        return self.interactions

    def generate_watch_history(self) -> pd.DataFrame:
        self.watch_history = pd.DataFrame({
            "user_id": [1, 2],
            "item_id": [1, 2],
            "watch_seconds": [1200, 900],
        })
        return self.watch_history

    def generate_reviews(self) -> pd.DataFrame:
        self.reviews = pd.DataFrame({
            "user_id": [1, 2],
            "item_id": [1, 2],
            "rating": [4.8, 4.3],
            "sentiment": ["positive", "neutral"],
        })
        return self.reviews

    def generate_retrieval_dataset(self) -> dict[str, pd.DataFrame]:
        if self.users is None:
            self.generate_users()
        if self.items is None:
            self.generate_items()
        if self.affinities is None:
            self.generate_affinities()

        retriever = RetrievalGenerator(
            self.foundation,
            users=self.users,
            items=self.items,
            affinity_df=self.affinities,
        )
        return retriever.generate(
            users=self.users,
            items=self.items,
            affinity_df=self.affinities,
        )

    def generate_ranking_dataset(self) -> pd.DataFrame:
        if self.users is None:
            self.generate_users()
        if self.items is None:
            self.generate_items()
        if self.affinities is None:
            self.generate_affinities()

        ranker = RankingGenerator(
            self.foundation,
            users=self.users,
            items=self.items,
            affinity_df=self.affinities,
        )
        self.ranking_df = ranker.generate(
            users=self.users,
            items=self.items,
            affinity_df=self.affinities,
        )
        return self.ranking_df

    def generate_re_ranking_dataset(self) -> pd.DataFrame:
        if self.ranking_df is None:
            self.generate_ranking_dataset()
        reranker = ReRankingGenerator(
            self.foundation,
            ranking_df=self.ranking_df,
            items=self.items,
        )
        self.reranking_df = reranker.generate(
            ranking_df=self.ranking_df,
            items=self.items,
        )
        return self.reranking_df

    def generate_search_dataset(self) -> pd.DataFrame:
        if self.users is None:
            self.generate_users()
        if self.items is None:
            self.generate_items()
        if self.affinities is None:
            self.generate_affinities()

        searcher = SearchGenerator(
            self.foundation,
            users=self.users,
            items=self.items,
            affinity_df=self.affinities,
        )
        return searcher.generate(
            users=self.users,
            items=self.items,
            affinity_df=self.affinities,
        )

    def generate_bandit_dataset(self) -> pd.DataFrame:
        if self.users is None:
            self.generate_users()
        if self.items is None:
            self.generate_items()
        if self.affinities is None:
            self.generate_affinities()

        bandit = BanditGenerator(
            self.foundation,
            users=self.users,
            items=self.items,
            affinity_df=self.affinities,
        )
        return bandit.generate(
            users=self.users,
            items=self.items,
            affinity_df=self.affinities,
        )

    def generate_feature_store(self) -> dict[str, pd.DataFrame]:
        if self.users is None:
            self.generate_users()
        if self.items is None:
            self.generate_items()
        if self.contexts is None:
            self.generate_contexts()
        if self.affinities is None:
            self.generate_affinities()

        builder = FeatureStoreBuilder(
            self.foundation,
            users=self.users,
            items=self.items,
            contexts=self.contexts,
            affinity_df=self.affinities,
        )
        self.feature_store = builder.generate(
            users=self.users,
            items=self.items,
            contexts=self.contexts,
            affinity_df=self.affinities,
        )
        return self.feature_store

    def generate_monitoring(self) -> dict[str, pd.DataFrame]:
        monitor = MonitoringGenerator(self.foundation)
        self.monitoring = monitor.generate()
        return self.monitoring

    def generate_business_metrics(self) -> dict[str, pd.DataFrame]:
        metrics = pd.DataFrame({
            "metric_name": ["DAU", "WAU", "MAU", "Retention", "Churn", "LTV", "ARPU", "Engagement", "Revenue"],
            "metric_value": [12000, 42000, 98000, 0.72, 0.08, 4.8, 5.9, 0.61, 480000],
        })
        business = BusinessMetricsGenerator(self.foundation, metrics=metrics)
        self.business_metrics = business.generate(metrics=metrics)
        return self.business_metrics

    def generate_drift(self) -> pd.DataFrame:
        metrics = pd.DataFrame({
            "metric_name": ["CTR", "Watch Rate", "Completion Rate", "Retention"],
            "metric_value": [0.08, 0.35, 0.52, 0.74],
        })
        drift = DriftGenerator(self.foundation, metrics=metrics)
        return drift.generate(metrics=metrics)

    def generate_session_sequences(self) -> dict[str, pd.DataFrame]:
        if self.users is None:
            self.generate_users()
        if self.items is None:
            self.generate_items()

        seq = SequentialRecommendationGenerator(
            self.foundation,
            users=self.users,
            items=self.items,
        )
        return seq.generate()

    def generate_graph(self) -> dict[str, pd.DataFrame]:
        if self.users is None:
            self.generate_users()
        if self.items is None:
            self.generate_items()

        graph = GraphRecommendationBuilder(
            self.foundation,
            users=self.users,
            items=self.items,
            interactions=pd.DataFrame({
                "user_id": [1, 2, 3],
                "item_id": [1, 2, 3],
                "interaction_weight": [0.9, 0.7, 0.8],
            }),
        )
        return graph.generate(
            users=self.users,
            items=self.items,
            interactions=pd.DataFrame({
                "user_id": [1, 2, 3],
                "item_id": [1, 2, 3],
                "interaction_weight": [0.9, 0.7, 0.8],
            }),
        )

    def generate_cold_start(self) -> pd.DataFrame:
        users = self.generate_users()
        items = self.generate_items()
        cold = ColdStartEngine(
            self.foundation,
            users=users,
            items=items,
        )
        return cold.generate(users=users, items=items)

    def generate_all(self) -> dict[str, Any]:
        return {
            "users": self.generate_users(),
            "items": self.generate_items(),
            "contexts": self.generate_contexts(),
            "affinities": self.generate_affinities(),
            "sessions": self.generate_sessions(),
            "interactions": self.generate_interactions(),
            "watch_history": self.generate_watch_history(),
            "reviews": self.generate_reviews(),
            "retrieval": self.generate_retrieval_dataset(),
            "ranking": self.generate_ranking_dataset(),
            "reranking": self.generate_re_ranking_dataset(),
            "search": self.generate_search_dataset(),
            "bandit": self.generate_bandit_dataset(),
            "feature_store": self.generate_feature_store(),
            "monitoring": self.generate_monitoring(),
            "business_metrics": self.generate_business_metrics(),
            "drift": self.generate_drift(),
            "sequential": self.generate_session_sequences(),
            "graph": self.generate_graph(),
            "cold_start": self.generate_cold_start(),
        }

    def export_all(
        self,
        output_dir: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Export all generated asset bundles to Parquet files.
        """

        output = self.generate_all()
        export_dir = Path(output_dir or self.config.output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        exported: dict[str, str] = {}

        def persist(name: str, frame: pd.DataFrame) -> None:
            target = export_dir / f"{name}.parquet"
            frame.to_parquet(target, index=False)
            exported[name] = str(target)

        for name, value in output.items():
            if isinstance(value, pd.DataFrame):
                persist(name, value)
            elif isinstance(value, dict):
                for subname, frame in value.items():
                    if isinstance(frame, pd.DataFrame):
                        persist(subname, frame)

        return exported


if __name__ == "__main__":
    generator = RecommendationDataGenerator()
    output = generator.generate_all()
    exported = generator.export_all()
    print("Generated asset keys:", list(output.keys()))
    print("Exported parquet files:", list(exported.keys()))
    print(output["users"].head())
    print(output["feature_store"]["user_features"].head())
