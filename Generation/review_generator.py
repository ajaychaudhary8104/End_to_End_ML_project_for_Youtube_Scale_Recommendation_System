from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class ReviewGenerator:
    """
    Phase 12 — Review Generator

    Creates synthetic user feedback records.

    Outputs:
    - rating
    - review_text
    - sentiment
    - helpfulness
    """

    SENTIMENTS = [
        "positive",
        "neutral",
        "negative"
    ]

    REVIEW_TEMPLATES = {
        "positive": [
            "Loved the experience and would watch again.",
            "Great recommendation with a smooth and engaging flow.",
            "High quality content and excellent match for my taste."
        ],
        "neutral": [
            "A decent watch overall with some enjoyable moments.",
            "It was fine, not memorable but still acceptable.",
            "Reasonable recommendation with a few standout elements."
        ],
        "negative": [
            "The recommendation did not fit my interests well.",
            "It was slow and not very engaging.",
            "A disappointing watch that did not meet expectations."
        ]
    }

    def __init__(
        self,
        foundation,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        interactions: Optional[pd.DataFrame] = None
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.users = users
        self.items = items
        self.interactions = interactions

    def _resolve_users(self, users: Optional[pd.DataFrame]) -> pd.DataFrame:
        if users is None:
            if self.users is None:
                raise ValueError("users must be provided")
            users = self.users

        working = users.copy()
        if "user_id" not in working.columns:
            working["user_id"] = np.arange(1, len(working) + 1)

        return working

    def _resolve_items(self, items: Optional[pd.DataFrame]) -> pd.DataFrame:
        if items is None:
            if self.items is None:
                items = pd.DataFrame({
                    "item_id": np.arange(1, self.config.n_items + 1)
                })
            else:
                items = self.items

        working = items.copy()
        if "item_id" not in working.columns:
            working["item_id"] = np.arange(1, len(working) + 1)

        return working

    def _resolve_interactions(self, interactions: Optional[pd.DataFrame]) -> pd.DataFrame:
        if interactions is None:
            if self.interactions is None:
                interactions = pd.DataFrame({
                    "user_id": self.rng.integers(1, self.config.n_users + 1, size=100),
                    "item_id": self.rng.integers(1, self.config.n_items + 1, size=100),
                    "watch": self.rng.integers(0, 2, size=100)
                })
            else:
                interactions = self.interactions

        working = interactions.copy()
        if "watch" not in working.columns:
            working["watch"] = self.rng.integers(0, 2, size=len(working))

        return working

    def _generate_rating(self, watch_flag: int) -> float:
        if watch_flag:
            base = self.rng.uniform(3.0, 5.0)
        else:
            base = self.rng.uniform(1.0, 3.5)

        return float(np.clip(np.round(base, 1), 1.0, 5.0))

    def _generate_sentiment(self, rating: float) -> str:
        if rating >= 4.0:
            return "positive"
        if rating <= 2.0:
            return "negative"
        return "neutral"

    def _generate_helpfulness(self, sentiment: str) -> float:
        if sentiment == "positive":
            return float(np.clip(self.rng.beta(5, 2), 0.0, 1.0))
        if sentiment == "negative":
            return float(np.clip(self.rng.beta(2, 5), 0.0, 1.0))
        return float(np.clip(self.rng.beta(3, 3), 0.0, 1.0))

    def _generate_review_text(self, sentiment: str) -> str:
        templates = self.REVIEW_TEMPLATES[sentiment]
        return self.rng.choice(templates)

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        interactions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate the synthetic review dataset.
        """

        users = self._resolve_users(users)
        items = self._resolve_items(items)
        interactions = self._resolve_interactions(interactions)

        reviews = []

        for _, row in interactions.iterrows():
            rating = self._generate_rating(int(row.get("watch", 0)))
            sentiment = self._generate_sentiment(rating)
            helpfulness = self._generate_helpfulness(sentiment)
            review_text = self._generate_review_text(sentiment)

            reviews.append({
                "user_id": int(row["user_id"]),
                "item_id": int(row["item_id"]),
                "rating": rating,
                "review_text": review_text,
                "sentiment": sentiment,
                "helpfulness": helpfulness
            })

        return pd.DataFrame(reviews)


class Phase12ReviewLayer:
    """
    Orchestrator-style wrapper for Phase 12.
    """

    def __init__(
        self,
        foundation,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: pd.DataFrame
    ):
        self.foundation = foundation
        self.users = users
        self.items = items
        self.interactions = interactions
        self.engine = ReviewGenerator(
            foundation,
            users=users,
            items=items,
            interactions=interactions
        )

    def generate(self) -> pd.DataFrame:
        return self.engine.generate(
            users=self.users,
            items=self.items,
            interactions=self.interactions
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

    users = pd.DataFrame({"user_id": [1, 2, 3]})
    items = pd.DataFrame({"item_id": [1, 2, 3]})
    interactions = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": [1, 2, 3],
        "watch": [1, 1, 0]
    })

    generator = ReviewGenerator(
        foundation,
        users=users,
        items=items,
        interactions=interactions
    )

    result = generator.generate(users, items, interactions)
    print(result.head())
