from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class SessionTimelineGenerator:
    """
    Phase 8 — Session Timeline Generator

    Creates realistic session timelines for recommendation users.
    Each synthetic session contains:

    - session_id
    - user_id
    - session_start
    - session_end
    - session_length
    - session_events
    - event_count

    The event stream follows a realistic progression:

    browse -> search -> click -> watch -> skip/review/share
    """

    EVENT_TYPES = [
        "browse",
        "search",
        "click",
        "watch",
        "skip",
        "review",
        "share"
    ]

    def __init__(
        self,
        foundation,
        users: Optional[pd.DataFrame] = None,
        contexts: Optional[pd.DataFrame] = None
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.users = users
        self.contexts = contexts

    def _resolve_users(
        self,
        users: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Resolve the user table for generation.
        """

        if users is None:
            if self.users is None:
                raise ValueError("users must be provided")
            users = self.users

        if "user_id" not in users.columns:
            users = users.copy()
            users["user_id"] = np.arange(1, len(users) + 1)

        return users

    def _resolve_contexts(
        self,
        contexts: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Resolve context rows to provide timestamp anchors.
        """

        if contexts is None:
            if self.contexts is None:
                contexts = pd.DataFrame(
                    {
                        "timestamp": pd.date_range(
                            start=self.config.start_date,
                            end=self.config.end_date,
                            freq="H"
                        )
                    }
                )
            else:
                contexts = self.contexts

        if "timestamp" not in contexts.columns:
            contexts["timestamp"] = pd.date_range(
                start=self.config.start_date,
                end=self.config.end_date,
                freq="H"
            )

        return contexts

    def _generate_session_count(
        self,
        user_row: pd.Series
    ) -> int:
        """
        Sample a plausible number of sessions per user.
        """

        if "session_frequency" in user_row.index:
            return max(
                1,
                int(user_row["session_frequency"])
            )

        return int(
            self.rng.integers(
                1,
                10
            )
        )

    def _sample_event_sequence(
        self,
        session_length_seconds: int
    ) -> list[str]:
        """
        Build a lightweight event stream for the session.
        """

        if session_length_seconds <= 60:
            event_count = self.rng.integers(1, 4)
        elif session_length_seconds <= 600:
            event_count = self.rng.integers(2, 7)
        else:
            event_count = self.rng.integers(4, 11)

        sequence = []
        first = self.rng.choice(
            ["browse", "search", "click"]
        )
        sequence.append(first)

        for _ in range(event_count - 1):
            next_event = self.rng.choice(
                self.EVENT_TYPES,
                p=[0.22, 0.18, 0.20, 0.16, 0.08, 0.08, 0.08]
            )
            sequence.append(next_event)

        return sequence

    def _session_duration_seconds(
        self,
        session_events: list[str]
    ) -> int:
        """
        Compute a plausible duration from the number of events.
        """

        base = len(session_events) * 35
        jitter = int(self.rng.integers(0, 180))
        return max(30, base + jitter)

    def generate(
        self,
        users: Optional[pd.DataFrame] = None,
        contexts: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate the full session timeline dataset.
        """

        users = self._resolve_users(users)
        contexts = self._resolve_contexts(contexts)

        rows = []
        session_counter = 1

        for _, user in users.iterrows():
            n_sessions = self._generate_session_count(user)

            for _ in range(n_sessions):
                event_sequence = self._sample_event_sequence(
                    session_length_seconds=300
                )
                session_length = self._session_duration_seconds(
                    event_sequence
                )

                start_time = contexts["timestamp"].sample(
                    1,
                    random_state=int(
                        self.rng.integers(
                            0,
                            1_000_000
                        )
                    )
                ).iloc[0]

                end_time = start_time + pd.to_timedelta(
                    session_length,
                    unit="s"
                )

                rows.append(
                    {
                        "session_id": f"S{session_counter:09d}",
                        "user_id": int(user["user_id"]),
                        "session_start": start_time,
                        "session_end": end_time,
                        "session_length": session_length,
                        "session_events": event_sequence,
                        "event_count": len(event_sequence),
                        "first_event": event_sequence[0],
                        "last_event": event_sequence[-1],
                    }
                )

                session_counter += 1

        return pd.DataFrame(rows)


class Phase8SessionTimelineLayer:
    """
    Orchestrator-style wrapper for Phase 8.
    """

    def __init__(
        self,
        foundation,
        users: pd.DataFrame,
        contexts: Optional[pd.DataFrame] = None
    ):
        self.foundation = foundation
        self.users = users
        self.contexts = contexts
        self.engine = SessionTimelineGenerator(
            foundation,
            users=users,
            contexts=contexts
        )

    def generate(self) -> pd.DataFrame:
        """
        Execute the full Phase-8 session-timeline generation.
        """

        return self.engine.generate(
            users=self.users,
            contexts=self.contexts
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

    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "session_frequency": [4, 6, 3]
        }
    )

    contexts = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2024-01-01",
                periods=10,
                freq="H"
            )
        }
    )

    generator = SessionTimelineGenerator(
        foundation,
        users=users,
        contexts=contexts
    )

    result = generator.generate(users, contexts)

    print(result.head(5))
