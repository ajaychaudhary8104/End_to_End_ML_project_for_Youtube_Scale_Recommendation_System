from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class ContextBiasEngine:
    """
    Phase 7 — Context Bias Engine

    Models context-specific behavior shifts in recommendation affinity.
    The engine injects a context-aware bias based on the local context
    around a recommendation event and then returns an adjusted affinity
    signal for downstream ranking and interaction generation.

    Expected behavior modifiers include:

    - Weekend effect
    - Holiday effect
    - Mobile effect
    - Prime-time effect
    - Campaign effect

    Outputs:
    - context_bias
    - adjusted_affinity
    """

    def __init__(
        self,
        foundation,
        contexts: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.contexts = contexts
        self.affinity_df = affinity_df

    def _coerce_inputs(
        self,
        contexts: pd.DataFrame,
        affinity_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate and normalize input tables.
        """

        if contexts is None or contexts.empty:
            raise ValueError("contexts must be a non-empty DataFrame")

        if affinity_df is None or affinity_df.empty:
            raise ValueError("affinity_df must be a non-empty DataFrame")

        working_contexts = contexts.copy()
        working_affinity = affinity_df.copy()

        if "context_id" not in working_contexts.columns:
            working_contexts["context_id"] = np.arange(
                len(working_contexts)
            )

        if "context_id" not in working_affinity.columns:
            if "context_id" in working_contexts.columns:
                working_affinity["context_id"] = (
                    self.rng.integers(
                        0,
                        len(working_contexts),
                        size=len(working_affinity)
                    )
                )

        if "device_type" not in working_contexts.columns:
            working_contexts["device_type"] = self.rng.choice(
                ["Mobile", "Desktop", "Tablet", "SmartTV"],
                size=len(working_contexts)
            )

        if "hour_of_day" not in working_contexts.columns:
            working_contexts["hour_of_day"] = self.rng.integers(
                0,
                24,
                size=len(working_contexts)
            )

        if "day_of_week" not in working_contexts.columns:
            working_contexts["day_of_week"] = self.rng.integers(
                0,
                7,
                size=len(working_contexts)
            )

        if "month" not in working_contexts.columns:
            working_contexts["month"] = self.rng.integers(
                1,
                13,
                size=len(working_contexts)
            )

        if "season" not in working_contexts.columns:
            working_contexts["season"] = self.rng.choice(
                ["winter", "spring", "summer", "autumn"],
                size=len(working_contexts)
            )

        if "traffic_source" not in working_contexts.columns:
            working_contexts["traffic_source"] = self.rng.choice(
                ["organic", "push_notification", "email", "advertisement"],
                size=len(working_contexts)
            )

        return working_contexts, working_affinity

    def _weekend_effect(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Weekend sessions have higher browsing/consumption propensity.
        """

        day_of_week = contexts["day_of_week"].to_numpy()
        effect = np.zeros(len(contexts), dtype=np.float32)

        weekend_mask = np.isin(day_of_week, [5, 6])
        effect[weekend_mask] = self.rng.uniform(
            0.10,
            0.35,
            size=np.sum(weekend_mask)
        )

        return effect

    def _holiday_effect(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Holiday periods amplify engagement and exploration behavior.
        """

        month = contexts["month"].to_numpy()
        effect = np.zeros(len(contexts), dtype=np.float32)

        holiday_mask = np.isin(month, [11, 12, 1])
        effect[holiday_mask] = self.rng.uniform(
            0.05,
            0.30,
            size=np.sum(holiday_mask)
        )

        return effect

    def _mobile_effect(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Mobile users tend to show shorter sessions and more shallow
        browsing, so the bias shifts toward quick-consumption content.
        """

        device = contexts["device_type"].to_numpy()
        effect = np.zeros(len(contexts), dtype=np.float32)

        mobile_mask = device == "Mobile"
        effect[mobile_mask] = self.rng.uniform(
            -0.12,
            0.08,
            size=np.sum(mobile_mask)
        )

        return effect

    def _prime_time_effect(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Prime-time recommendation sessions receive an engagement boost.
        """

        hour = contexts["hour_of_day"].to_numpy()
        effect = np.zeros(len(contexts), dtype=np.float32)

        prime_mask = np.logical_and(hour >= 18, hour <= 22)
        effect[prime_mask] = self.rng.uniform(
            0.08,
            0.28,
            size=np.sum(prime_mask)
        )

        return effect

    def _campaign_effect(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Push campaigns and email-triggered sessions increase browsing
        impulse and recommendation sensitivity.
        """

        traffic = contexts["traffic_source"].to_numpy()
        effect = np.zeros(len(contexts), dtype=np.float32)

        campaign_mask = np.isin(
            traffic,
            ["push_notification", "email", "advertisement"]
        )

        effect[campaign_mask] = self.rng.uniform(
            0.05,
            0.22,
            size=np.sum(campaign_mask)
        )

        return effect

    def _context_bias(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Aggregate context-specific modifiers into a scalar bias term.
        """

        bias = (
            self._weekend_effect(contexts)
            + self._holiday_effect(contexts)
            + self._mobile_effect(contexts)
            + self._prime_time_effect(contexts)
            + self._campaign_effect(contexts)
        )

        return np.clip(
            bias,
            -0.25,
            0.55
        )

    def _resolve_affinity_signal(
        self,
        affinity_df: pd.DataFrame
    ) -> pd.Series:
        """
        Resolve the underlying score column that should be adjusted.
        """

        candidates = [
            "adjusted_affinity",
            "candidate_generation_score",
            "ranking_label",
            "bandit_reward",
            "positive_interaction_label",
            "high_value_label",
            "context_affinity",
            "user_item_embedding_affinity",
            "click_probability",
            "watch_probability",
            "completion_probability",
            "satisfaction_probability",
            "retention_probability"
        ]

        for column in candidates:
            if column in affinity_df.columns:
                return affinity_df[column]

        return pd.Series(
            np.full(
                len(affinity_df),
                0.5,
                dtype=np.float32
            ),
            index=affinity_df.index
        )

    def _join_contexts(
        self,
        contexts: pd.DataFrame,
        affinity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join context rows to each affinity row using context_id when
        available, otherwise use a random sampled row per affinity.
        """

        merged = affinity_df.copy()

        if "context_id" in merged.columns and "context_id" in contexts.columns:
            merged = merged.merge(
                contexts,
                on="context_id",
                how="left"
            )
            return merged

        if "context_id" not in merged.columns:
            merged["context_id"] = self.rng.integers(
                0,
                len(contexts),
                size=len(merged)
            )

        merged = merged.merge(
            contexts,
            on="context_id",
            how="left"
        )

        return merged

    def generate(
        self,
        contexts: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Apply context-aware biasing and return a context-adjusted
        affinity dataset.
        """

        if contexts is None:
            contexts = self.contexts

        if affinity_df is None:
            affinity_df = self.affinity_df

        working_contexts, working_affinity = self._coerce_inputs(
            contexts,
            affinity_df
        )

        merged = self._join_contexts(
            working_contexts,
            working_affinity
        )

        bias = self._context_bias(merged)

        base_signal = self._resolve_affinity_signal(merged)
        base_signal = base_signal.astype(np.float32)

        adjusted = np.clip(
            base_signal.to_numpy() + bias,
            0.0,
            1.0
        )

        merged["context_bias"] = bias
        merged["adjusted_affinity"] = adjusted

        return merged


class Phase7ContextBiasLayer:
    """
    Orchestrator-style wrapper for Phase 7.
    """

    def __init__(
        self,
        foundation,
        contexts: pd.DataFrame,
        affinity_df: pd.DataFrame
    ):
        self.foundation = foundation
        self.contexts = contexts
        self.affinity_df = affinity_df
        self.engine = ContextBiasEngine(
            foundation,
            contexts=contexts,
            affinity_df=affinity_df
        )

    def generate(self) -> pd.DataFrame:
        """
        Execute the complete Phase-7 context-bias pipeline.
        """

        return self.engine.generate(
            contexts=self.contexts,
            affinity_df=self.affinity_df
        )


if __name__ == "__main__":
    from foundation import FoundationLayer, GeneratorConfig

    config = GeneratorConfig(
        n_users=1_000,
        n_items=100,
        embedding_dim=32,
        random_state=42
    )

    foundation = FoundationLayer(config)

    contexts = pd.DataFrame(
        {
            "context_id": [0, 1, 2, 3],
            "device_type": ["Mobile", "Desktop", "Mobile", "SmartTV"],
            "hour_of_day": [17, 21, 2, 20],
            "day_of_week": [5, 6, 1, 4],
            "month": [12, 11, 8, 6],
            "season": ["winter", "winter", "summer", "summer"],
            "traffic_source": ["email", "advertisement", "organic", "push_notification"]
        }
    )

    affinity = pd.DataFrame(
        {
            "context_id": [0, 1, 2, 3],
            "candidate_generation_score": [0.35, 0.60, 0.55, 0.40]
        }
    )

    engine = ContextBiasEngine(foundation, contexts, affinity)
    result = engine.generate(contexts, affinity)

    print(result[["context_id", "context_bias", "adjusted_affinity"]])
