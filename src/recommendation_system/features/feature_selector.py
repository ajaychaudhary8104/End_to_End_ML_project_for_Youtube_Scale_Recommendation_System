from __future__ import annotations

import pandas as pd


class FeatureSelector:
    def __init__(self, exclude: set[str] | None = None):
        self.exclude = exclude or set()

    def select(self, frame: pd.DataFrame, target: str | None = None, max_features: int | None = None) -> pd.DataFrame:
        if frame is None or frame.empty:
            raise ValueError("frame must be a non-empty DataFrame")
        excluded = self.exclude | ({target} if target else set())
        columns = [column for column in frame.columns if column not in excluded and frame[column].nunique(dropna=False) > 1]
        if max_features is not None:
            columns = columns[:max_features]
        selected = frame[columns].copy()
        if target and target in frame:
            selected[target] = frame[target]
        return selected
