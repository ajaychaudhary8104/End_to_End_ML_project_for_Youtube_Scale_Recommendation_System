from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class BusinessMetricsGenerator:
    """
    Phase 24 — Business Metrics Generator

    Generates executive KPI-style outputs for:
    - business_metrics
    - growth_metrics
    - retention_metrics
    - revenue_metrics
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
                    "metric_name": ["DAU", "WAU", "MAU", "Retention", "Churn", "LTV", "ARPU", "Engagement", "Revenue"],
                    "metric_value": [12000, 42000, 98000, 0.72, 0.08, 4.8, 5.9, 0.61, 480000],
                })
            else:
                metrics = self.metrics

        working = metrics.copy()
        if "metric_name" not in working.columns:
            working["metric_name"] = [f"metric_{i}" for i in range(1, len(working) + 1)]
        if "metric_value" not in working.columns:
            working["metric_value"] = self.rng.random(len(working))

        return working

    def generate_business_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        business = metrics[metrics["metric_name"].isin(["DAU", "WAU", "MAU", "Engagement", "Revenue"])].copy()
        business["metric_group"] = "business_metrics"
        return business

    def generate_growth_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        growth = metrics[metrics["metric_name"].isin(["DAU", "WAU", "MAU"])].copy()
        growth["metric_group"] = "growth_metrics"
        return growth

    def generate_retention_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        retention = metrics[metrics["metric_name"].isin(["Retention", "Churn", "LTV"])].copy()
        retention["metric_group"] = "retention_metrics"
        return retention

    def generate_revenue_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        revenue = metrics[metrics["metric_name"].isin(["ARPU", "Revenue"])].copy()
        revenue["metric_group"] = "revenue_metrics"
        return revenue

    def generate(self, metrics: Optional[pd.DataFrame] = None) -> dict[str, pd.DataFrame]:
        """
        Generate executive KPI tables.
        """

        metrics = self._resolve_metrics(metrics)
        return {
            "business_metrics": self.generate_business_metrics(metrics),
            "growth_metrics": self.generate_growth_metrics(metrics),
            "retention_metrics": self.generate_retention_metrics(metrics),
            "revenue_metrics": self.generate_revenue_metrics(metrics),
        }


class Phase24BusinessMetricsLayer:
    """
    Orchestrator-style wrapper for Phase 24.
    """

    def __init__(self, foundation, metrics: Optional[pd.DataFrame] = None):
        self.foundation = foundation
        self.metrics = metrics
        self.engine = BusinessMetricsGenerator(foundation, metrics=metrics)

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
        "metric_name": ["DAU", "WAU", "MAU", "Retention", "Churn", "LTV", "ARPU", "Engagement", "Revenue"],
        "metric_value": [12000, 42000, 98000, 0.72, 0.08, 4.8, 5.9, 0.61, 480000],
    })

    generator = BusinessMetricsGenerator(foundation, metrics=metrics)
    result = generator.generate(metrics)
    print(result["business_metrics"].head())
    print(result["growth_metrics"].head())
    print(result["retention_metrics"].head())
    print(result["revenue_metrics"].head())
