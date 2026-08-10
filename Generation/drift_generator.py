from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class DriftGenerator:
    """
    Phase 22 — Drift Generator

    Creates production drift scenarios with:
    - drift_score
    - drift_type
    """

    def __init__(
        self,
        foundation,
        metrics: Optional[pd.DataFrame] = None,
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.metrics = metrics

    def _resolve_metrics(self, metrics: Optional[pd.DataFrame]) -> pd.DataFrame:
        if metrics is None:
            if self.metrics is None:
                metrics = pd.DataFrame({
                    "metric_name": ["CTR", "Watch Rate", "Completion Rate", "Retention"],
                    "metric_value": [0.08, 0.35, 0.52, 0.74],
                })
            else:
                metrics = self.metrics

        working = metrics.copy()
        if "metric_name" not in working.columns:
            working["metric_name"] = [f"metric_{i}" for i in range(1, len(working) + 1)]
        if "metric_value" not in working.columns:
            working["metric_value"] = self.rng.random(len(working))

        return working

    def generate(
        self,
        metrics: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate a drift dataset with a score and drift type label.
        """

        metrics = self._resolve_metrics(metrics)
        drift_rows = []

        for _, row in metrics.iterrows():
            value = float(row["metric_value"])
            drift_score = float(np.clip(abs(value - 0.5) * 2.0, 0.0, 1.0))

            if drift_score > 0.75:
                drift_type = "catalog drift"
            elif drift_score > 0.50:
                drift_type = "seasonality drift"
            elif drift_score > 0.25:
                drift_type = "popularity drift"
            else:
                drift_type = "preference drift"

            drift_rows.append({
                "metric_name": row["metric_name"],
                "drift_score": drift_score,
                "drift_type": drift_type,
            })

        return pd.DataFrame(drift_rows)


class Phase22DriftLayer:
    """
    Orchestrator-style wrapper for Phase 22.
    """

    def __init__(self, foundation, metrics: Optional[pd.DataFrame] = None):
        self.foundation = foundation
        self.metrics = metrics
        self.engine = DriftGenerator(foundation, metrics=metrics)

    def generate(self) -> pd.DataFrame:
        return self.engine.generate(metrics=self.metrics)


if __name__ == "__main__":
    from foundation import FoundationLayer, GeneratorConfig

    config = GeneratorConfig(
        n_users=10,
        n_items=10,
        embedding_dim=8,
        random_state=42,
    )

    foundation = FoundationLayer(config)
    metrics = pd.DataFrame({
        "metric_name": ["CTR", "Watch Rate", "Completion Rate", "Retention"],
        "metric_value": [0.08, 0.35, 0.52, 0.74],
    })

    generator = DriftGenerator(foundation, metrics=metrics)
    result = generator.generate(metrics)
    print(result.head())
