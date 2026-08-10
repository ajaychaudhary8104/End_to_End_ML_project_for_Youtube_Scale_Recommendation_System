from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class InteractionGenerator:
    """
    Phase 9 — Interaction Generator

    Generates synthetic recommendation interaction events based on
    session timelines and affinity signals.

    Output fields include:

    - impression
    - click
    - watch
    - completion
    - like
    - dislike
    - share
    - save

    This generator is intentionally lightweight but realistic: it
    creates per-session candidate interaction records with session-level
    behavioral semantics and outcome flags.
    """

    EVENT_COLUMNS = [
        "impression",
        "click",
        "watch",
        "completion",
        "like",
        "dislike",
        "share",
        "save"
    ]

    def __init__(
        self,
        foundation,
        sessions: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.sessions = sessions
        self.affinity_df = affinity_df
        self.items = items

    def _resolve_sessions(
        self,
        sessions: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if sessions is None:
            if self.sessions is None:
                raise ValueError("sessions must be provided")
            sessions = self.sessions

        working = sessions.copy()

        if "session_id" not in working.columns:
            working["session_id"] = [
                f"S{i:09d}" for i in range(1, len(working) + 1)
            ]

        if "user_id" not in working.columns:
            working["user_id"] = self.rng.integers(
                1,
                self.config.n_users + 1,
                size=len(working)
            )

        if "session_length" not in working.columns:
            working["session_length"] = self.rng.integers(
                30,
                1800,
                size=len(working)
            )

        return working

    def _resolve_affinity(
        self,
        affinity_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if affinity_df is None:
            if self.affinity_df is None:
                affinity_df = pd.DataFrame(
                    {
                        "user_id": self.rng.integers(
                            1,
                            self.config.n_users + 1,
                            size=100
                        ),
                        "item_id": self.rng.integers(
                            1,
                            self.config.n_items + 1,
                            size=100
                        ),
                        "candidate_generation_score": self.rng.random(100)
                    }
                )
            else:
                affinity_df = self.affinity_df

        return affinity_df.copy()

    def _resolve_items(
        self,
        items: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if items is None:
            if self.items is None:
                items = pd.DataFrame(
                    {
                        "item_id": np.arange(1, self.config.n_items + 1)
                    }
                )
            else:
                items = self.items

        return items.copy()

    def _sample_items_for_session(
        self,
        session_row: pd.Series,
        item_pool: pd.DataFrame
    ) -> pd.Series:
        """
        Sample a candidate item that can be linked to the session.
        """

        n_candidates = min(3, len(item_pool))
        sampled = item_pool.sample(
            n=n_candidates,
            replace=False,
            random_state=int(
                self.rng.integers(
                    0,
                    1_000_000
                )
            )
        )

        return sampled.iloc[0]

    def _generate_event_flags(
        self,
        session_row: pd.Series,
        item_row: pd.Series
    ) -> dict[str, int]:
        """
        Probabilistic binary flag generation.
        """

        score = self.rng.random()
        click_prob = 0.40 + min(0.25, item_row.get("popularity_score", 0.5) * 0.1)
        watch_prob = max(0.15, click_prob * 0.70)
        completion_prob = max(0.10, watch_prob * 0.55)
        like_prob = max(0.05, completion_prob * 0.35)
        dislike_prob = max(0.01, (1.0 - completion_prob) * 0.12)
        share_prob = max(0.01, like_prob * 0.25)
        save_prob = max(0.02, like_prob * 0.40)

        flags = {
            "impression": 1,
            "click": int(score < click_prob),
            "watch": int(score < watch_prob),
            "completion": int(score < completion_prob),
            "like": int(score < like_prob),
            "dislike": int(score < dislike_prob),
            "share": int(score < share_prob),
            "save": int(score < save_prob),
        }

        return flags

    def generate(
        self,
        sessions: Optional[pd.DataFrame] = None,
        affinity_df: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate the interaction dataset.
        """

        sessions = self._resolve_sessions(sessions)
        affinity_df = self._resolve_affinity(affinity_df)
        items = self._resolve_items(items)

        rows = []

        for _, session in sessions.iterrows():
            item_row = self._sample_items_for_session(session, items)

            flags = self._generate_event_flags(session, item_row)

            rows.append(
                {
                    "session_id": session["session_id"],
                    "user_id": int(session["user_id"]),
                    "item_id": int(item_row["item_id"]),
                    "session_length": int(session["session_length"]),
                    **flags,
                }
            )

        interactions = pd.DataFrame(rows)

        if "click" in interactions.columns:
            interactions["click"] = interactions["click"].astype(np.int8)
        if "watch" in interactions.columns:
            interactions["watch"] = interactions["watch"].astype(np.int8)
        if "completion" in interactions.columns:
            interactions["completion"] = interactions["completion"].astype(np.int8)
        if "like" in interactions.columns:
            interactions["like"] = interactions["like"].astype(np.int8)
        if "dislike" in interactions.columns:
            interactions["dislike"] = interactions["dislike"].astype(np.int8)
        if "share" in interactions.columns:
            interactions["share"] = interactions["share"].astype(np.int8)
        if "save" in interactions.columns:
            interactions["save"] = interactions["save"].astype(np.int8)

        return interactions


class Phase9InteractionLayer:
    """
    Orchestrator-style wrapper for Phase 9.
    """

    def __init__(
        self,
        foundation,
        sessions: pd.DataFrame,
        affinity_df: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None
    ):
        self.foundation = foundation
        self.sessions = sessions
        self.affinity_df = affinity_df
        self.items = items
        self.engine = InteractionGenerator(
            foundation,
            sessions=sessions,
            affinity_df=affinity_df,
            items=items
        )

    def generate(self) -> pd.DataFrame:
        return self.engine.generate(
            sessions=self.sessions,
            affinity_df=self.affinity_df,
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

    sessions = pd.DataFrame(
        {
            "session_id": ["S000000001", "S000000002"],
            "user_id": [1, 2],
            "session_length": [320, 540]
        }
    )

    items = pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "popularity_score": [0.9, 0.7, 0.4]
        }
    )

    generator = InteractionGenerator(
        foundation,
        sessions=sessions,
        items=items
    )

    result = generator.generate(sessions, None, items)
    print(result.head())
