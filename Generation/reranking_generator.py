from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class ReRankingGenerator:
    """
    Phase 15 — Re-Ranking Generator

    Creates post-ranking optimization datasets with diversity,
    freshness, novelty, fairness, and coverage objectives.

    Outputs:
    - rerank_score
    - diversity_score
    - novelty_score
    """

    def __init__(
        self,
        foundation,
        ranking_df: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.ranking_df = ranking_df
        self.items = items

    def _resolve_ranking_df(
        self,
        ranking_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if ranking_df is None:
            if self.ranking_df is None:
                raise ValueError("ranking_df must be provided")
            ranking_df = self.ranking_df

        working = ranking_df.copy()
        if "rank_label" not in working.columns:
            working["rank_label"] = self.rng.random(len(working))

        return working

    def _resolve_items(
        self,
        items: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if items is None:
            if self.items is None:
                items = pd.DataFrame({
                    "item_id": np.arange(1, self.config.n_items + 1),
                    "freshness_score": self.rng.random(self.config.n_items),
                    "genre": self.rng.choice([
                        "Action", "Drama", "Comedy", "SciFi", "Sports"
                    ], size=self.config.n_items)
                })
            else:
                items = self.items

        working = items.copy()
        if "item_id" not in working.columns:
            working["item_id"] = np.arange(1, len(working) + 1)

        return working

    def _diversity_score(
        self,
        ranking_df: pd.DataFrame,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Penalize excessive topical repetition inside a query result set.
        """

        diversity_map = {}
        for query_id, query in ranking_df.groupby("query_id"):
            item_ids = query["item_id"].to_numpy()
            genres = items.loc[items["item_id"].isin(item_ids), "genre"].tolist()
            unique_genres = len(set(genres))
            diversity_map[query_id] = min(
                1.0,
                unique_genres / max(len(item_ids), 1)
            )

        diversity = ranking_df["query_id"].map(diversity_map)
        return diversity.to_numpy(dtype=np.float32)

    def _novelty_score(
        self,
        ranking_df: pd.DataFrame,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Reward items with stronger freshness / novelty.
        """

        novelty_map = {}
        for query_id, query in ranking_df.groupby("query_id"):
            item_ids = query["item_id"].to_numpy()
            freshness = items.loc[items["item_id"].isin(item_ids), "freshness_score"]
            novelty_map[query_id] = float(freshness.mean())

        novelty = ranking_df["query_id"].map(novelty_map)
        return novelty.to_numpy(dtype=np.float32)

    def generate(
        self,
        ranking_df: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate a reranking dataset.
        """

        ranking_df = self._resolve_ranking_df(ranking_df)
        items = self._resolve_items(items)

        diversity = self._diversity_score(ranking_df, items)
        novelty = self._novelty_score(ranking_df, items)

        base_scores = ranking_df["rank_label"].to_numpy(dtype=np.float32)
        rerank = (
            0.45 * base_scores
            + 0.35 * diversity
            + 0.20 * novelty
        )

        output = ranking_df.copy()
        output["rerank_score"] = np.clip(rerank, 0.0, 1.0)
        output["diversity_score"] = np.clip(diversity, 0.0, 1.0)
        output["novelty_score"] = np.clip(novelty, 0.0, 1.0)

        return output


class Phase15ReRankingLayer:
    """
    Orchestrator-style wrapper for Phase 15.
    """

    def __init__(
        self,
        foundation,
        ranking_df: pd.DataFrame,
        items: pd.DataFrame
    ):
        self.foundation = foundation
        self.ranking_df = ranking_df
        self.items = items
        self.engine = ReRankingGenerator(
            foundation,
            ranking_df=ranking_df,
            items=items
        )

    def generate(self) -> pd.DataFrame:
        return self.engine.generate(
            ranking_df=self.ranking_df,
            items=self.items
        )


if __name__ == "__main__":
    from foundation import FoundationLayer, GeneratorConfig

    config = GeneratorConfig(
        n_users=100,
        n_items=10,
        embedding_dim=32,
        random_state=42
    )

    foundation = FoundationLayer(config)

    ranking_df = pd.DataFrame({
        "query_id": ["q000000001", "q000000001", "q000000001"],
        "user_id": [1, 1, 1],
        "item_id": [1, 2, 3],
        "rank_label": [0.9, 0.7, 0.5]
    })

    items = pd.DataFrame({
        "item_id": [1, 2, 3],
        "freshness_score": [0.9, 0.6, 0.4],
        "genre": ["Action", "Action", "Drama"]
    })

    generator = ReRankingGenerator(foundation, ranking_df, items)
    result = generator.generate(ranking_df, items)
    print(result.head())
