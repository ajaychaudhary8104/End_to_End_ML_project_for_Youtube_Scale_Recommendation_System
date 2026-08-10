from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class PositionBiasEngine:
    """
    Phase 6 — Position Bias Engine

    Simulates the ranking-position effect present in recommendation
    systems, specifically the decay in attention, visibility, and
    downstream exposure probability as an item moves farther down a
    recommendation list.

    The engine is designed to consume a context dataframe that contains
    ranking surface metadata such as:

    - page_position
    - recommendation_slot
    - surface_type

    Optional context columns already supported by the pipeline include:

    - homepage_bias
    - search_bias

    The output is a compact dataframe with:

    - position_bias_score
    - visibility_score
    - exposure_probability
    """

    def __init__(
        self,
        foundation,
        contexts: Optional[pd.DataFrame] = None
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.contexts = contexts

    def _coerce_contexts(
        self,
        contexts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize input contexts and ensure the required ranking
        metadata exists.
        """

        if contexts is None or contexts.empty:
            raise ValueError(
                "contexts must be a non-empty pandas DataFrame"
            )

        working = contexts.copy()

        if "page_position" not in working.columns:
            working["page_position"] = self.rng.integers(
                1,
                101,
                size=len(working)
            )

        if "recommendation_slot" not in working.columns:
            working["recommendation_slot"] = self.rng.integers(
                1,
                51,
                size=len(working)
            )

        if "surface_type" not in working.columns:
            working["surface_type"] = self.rng.choice(
                [
                    "homepage",
                    "search",
                    "details_page",
                    "continue_watching",
                    "trending",
                    "recommended",
                    "category_page"
                ],
                size=len(working)
            )

        return working

    def _generate_position_bias_score(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Power-law decay based on the visible rank position.

        Lower ranks receive a sharply smaller attention weight.
        """

        positions = (
            contexts["page_position"]
            .clip(lower=1)
            .astype(np.float32)
            .to_numpy()
        )

        decay = self.rng.uniform(
            0.55,
            1.30,
            size=len(contexts)
        )

        score = np.power(
            positions,
            -decay
        )

        score /= np.max(score)

        return np.clip(
            score,
            0.0,
            1.0
        )

    def _generate_visibility_score(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Visibility depends on prompt surface and its physical slot.

        Higher slots on the homepage are more visible, while deep
        positions are partially masked or competing with other results.
        """

        positions = (
            contexts["page_position"]
            .clip(lower=1)
            .astype(np.float32)
            .to_numpy()
        )

        max_position = max(
            int(np.max(positions)),
            1
        )

        surface = contexts["surface_type"].to_numpy()

        visibility = 1.0 - (
            (positions - 1)
            / max_position
        ) * 0.90

        visibility = np.clip(
            visibility,
            0.05,
            1.0
        )

        homepage_mask = surface == "homepage"
        search_mask = surface == "search"
        recommended_mask = surface == "recommended"

        visibility[homepage_mask] += 0.05
        visibility[search_mask] -= 0.03
        visibility[recommended_mask] += 0.02

        return np.clip(
            visibility,
            0.05,
            1.0
        )

    def _generate_exposure_probability(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Combine position bias and visibility into a probability-like
        exposure estimate, optionally boosted by available surface bias
        columns.
        """

        position_bias = (
            self._generate_position_bias_score(
                contexts
            )
        )

        visibility = (
            self._generate_visibility_score(
                contexts
            )
        )

        homepage_bias = np.zeros(
            len(contexts),
            dtype=np.float32
        )
        search_bias = np.zeros(
            len(contexts),
            dtype=np.float32
        )

        if "homepage_bias" in contexts.columns:
            homepage_bias = contexts["homepage_bias"].to_numpy()

        if "search_bias" in contexts.columns:
            search_bias = contexts["search_bias"].to_numpy()

        surface_bias = np.where(
            contexts["surface_type"].to_numpy() == "search",
            search_bias,
            homepage_bias
        )

        exposure = (
            0.55 * position_bias
            + 0.35 * visibility
            + 0.10 * np.clip(surface_bias, 0.0, 1.0)
        )

        return np.clip(
            exposure,
            0.0,
            1.0
        )

    def generate(
        self,
        contexts: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate a position-bias dataset compatible with the
        downstream ranking / exposure layer.
        """

        if contexts is None:
            contexts = self.contexts

        working = self._coerce_contexts(contexts)

        output = pd.DataFrame(
            {
                "position_bias_score": (
                    self._generate_position_bias_score(
                        working
                    )
                ),
                "visibility_score": (
                    self._generate_visibility_score(
                        working
                    )
                ),
                "exposure_probability": (
                    self._generate_exposure_probability(
                        working
                    )
                )
            }
        )

        return output


class Phase6PositionBiasLayer:
    """
    Orchestrator-style wrapper for the Phase 6 engine.
    """

    def __init__(
        self,
        foundation,
        contexts: pd.DataFrame
    ):
        self.foundation = foundation
        self.contexts = contexts
        self.engine = PositionBiasEngine(
            foundation,
            contexts=contexts
        )

    def generate(self) -> pd.DataFrame:
        """
        Execute the full Phase-6 position bias generation.
        """

        return self.engine.generate(
            contexts=self.contexts
        )


if __name__ == "__main__":
    # Lightweight smoke test for the new engine.
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
            "page_position": [1, 2, 5, 10, 20, 50],
            "recommendation_slot": [1, 2, 3, 4, 5, 6],
            "surface_type": [
                "homepage",
                "recommended",
                "search",
                "details_page",
                "trending",
                "category_page"
            ],
            "homepage_bias": [0.8, 0.75, 0.4, 0.3, 0.25, 0.2],
            "search_bias": [0.2, 0.3, 0.7, 0.5, 0.4, 0.35]
        }
    )

    engine = PositionBiasEngine(foundation, contexts)
    result = engine.generate(contexts)

    print(result.head())
