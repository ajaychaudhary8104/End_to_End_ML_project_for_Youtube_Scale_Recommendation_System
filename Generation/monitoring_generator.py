from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class MonitoringGenerator:
    """
    Phase 23 — Monitoring Generator

    Creates daily, weekly, and real-time monitoring metrics.
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
                    "metric_name": ["CTR", "Watch Rate", "Completion Rate", "Retention", "Revenue"],
                    "metric_value": [0.08, 0.35, 0.52, 0.74, 12.5],
                })
            else:
                metrics = self.metrics

        working = metrics.copy()
        if "metric_name" not in working.columns:
            working["metric_name"] = [f"metric_{i}" for i in range(1, len(working) + 1)]
        if "metric_value" not in working.columns:
            working["metric_value"] = self.rng.random(len(working))

        return working

    def generate_daily_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in metrics.iterrows():
            rows.append({
                "metric_name": row["metric_name"],
                "metric_period": "daily",
                "metric_value": float(row["metric_value"]),
            })
        return pd.DataFrame(rows)

    def generate_weekly_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in metrics.iterrows():
            rows.append({
                "metric_name": row["metric_name"],
                "metric_period": "weekly",
                "metric_value": float(row["metric_value"] * 1.15),
            })
        return pd.DataFrame(rows)

    def generate_real_time_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in metrics.iterrows():
            rows.append({
                "metric_name": row["metric_name"],
                "metric_period": "real_time",
                "metric_value": float(np.clip(row["metric_value"] + self.rng.normal(0, 0.05), 0.0, None)),
            })
        return pd.DataFrame(rows)

    def generate(
        self,
        metrics: Optional[pd.DataFrame] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Build the three monitoring metric tables.
        """

        metrics = self._resolve_metrics(metrics)
        return {
            "daily_metrics": self.generate_daily_metrics(metrics),
            "weekly_metrics": self.generate_weekly_metrics(metrics),
            "real_time_metrics": self.generate_real_time_metrics(metrics),
        }


class Phase23MonitoringLayer:
    """
    Orchestrator-style wrapper for Phase 23.
    """

    def __init__(self, foundation, metrics: Optional[pd.DataFrame] = None):
        self.foundation = foundation
        self.metrics = metrics
        self.engine = MonitoringGenerator(foundation, metrics=metrics)

    def generate(self) -> dict[str, pd.DataFrame]:
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
        "metric_name": ["CTR", "Watch Rate", "Completion Rate", "Retention", "Revenue"],
        "metric_value": [0.08, 0.35, 0.52, 0.74, 12.5],
    })

    generator = MonitoringGenerator(foundation, metrics=metrics)
    result = generator.generate(metrics)
    print(result["daily_metrics"].head())
    print(result["weekly_metrics"].head())
    print(result["real_time_metrics"].head())
