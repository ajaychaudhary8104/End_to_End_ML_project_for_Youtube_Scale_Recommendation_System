from __future__ import annotations

import numpy as np
import pandas as pd


class AffinityEngine:
    """
    Production-grade user-item affinity engine.

    Produces latent preference scores used by
    every downstream recommendation dataset.

    Output:
        user_item_affinity
    """

    def __init__(
        self,
        foundation,
        users: pd.DataFrame,
        items: pd.DataFrame
    ):

        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng

        self.users = users
        self.items = items

    # ==================================================
    # GENRE AFFINITY
    # ==================================================

    def _genre_affinity(
        self,
        user_row: pd.Series,
        item_row: pd.Series
    ) -> float:
        """
        User genre preference match.
        """

        user_genres = set(
            user_row[
                "preferred_genres"
            ]
        )

        item_genres = set(
            item_row[
                "sub_genres"
            ]
        )

        overlap = len(
            user_genres.intersection(
                item_genres
            )
        )

        denominator = max(
            len(item_genres),
            1
        )

        return overlap / denominator

    # ==================================================
    # LANGUAGE AFFINITY
    # ==================================================

    def _language_affinity(
        self,
        user_row: pd.Series,
        item_row: pd.Series
    ) -> float:

        return float(
            user_row[
                "preferred_language"
            ]
            ==
            item_row[
                "language"
            ]
        )

    # ==================================================
    # POPULARITY AFFINITY
    # ==================================================

    def _popularity_affinity(
        self,
        user_row: pd.Series,
        item_row: pd.Series
    ) -> float:
        """
        Popularity preference matching.
        """

        user_pref = (
            user_row[
                "popularity_preference"
            ]
        )

        item_pop = (
            item_row[
                "popularity_score"
            ]
        )

        return (
            1.0
            -
            abs(
                user_pref
                -
                item_pop
            )
        )

    # ==================================================
    # NOVELTY AFFINITY
    # ==================================================

    def _novelty_affinity(
        self,
        user_row: pd.Series,
        item_row: pd.Series
    ) -> float:

        novelty = (
            item_row[
                "freshness_score"
            ]
        )

        preference = (
            user_row[
                "novelty_preference"
            ]
        )

        return (
            1.0
            -
            abs(
                novelty
                -
                preference
            )
        )

    # ==================================================
    # QUALITY AFFINITY
    # ==================================================

    def _quality_affinity(
        self,
        item_row: pd.Series
    ) -> float:

        return float(
            item_row[
                "quality_score"
            ]
        )

    # ==================================================
    # RECENCY AFFINITY
    # ==================================================

    def _recency_affinity(
        self,
        item_row: pd.Series
    ) -> float:

        return float(
            item_row[
                "freshness_score"
            ]
        )

    # ==================================================
    # CREATOR AFFINITY
    # ==================================================

    def _creator_affinity(
        self,
        user_row: pd.Series,
        item_row: pd.Series
    ) -> float:
        """
        Placeholder creator preference.

        Later versions use creator embeddings.
        """

        if (
            hash(
                item_row["creator"]
            )
            %
            100
        ) < 15:

            return self.rng.uniform(
                0.7,
                1.0
            )

        return self.rng.uniform(
            0.0,
            0.6
        )

    # ==================================================
    # USER EMBEDDING AFFINITY
    # ==================================================

    def _user_item_embedding_affinity(
        self,
        user_row: pd.Series,
        item_row: pd.Series
    ) -> float:
        """
        Latent preference similarity.

        Uses user latent factors and
        item content embeddings.
        """

        user_embedding = np.asarray(
            user_row["latent_vector"],
            dtype=np.float32
        )

        item_idx = (
            item_row.name
        )

        item_embedding = (
            self.items_content_embeddings[
                item_idx
            ]
        )

        similarity = np.dot(
            user_embedding,
            item_embedding
        )

        similarity /= (
            np.linalg.norm(
                user_embedding
            )
            *
            np.linalg.norm(
                item_embedding
            )
            +
            1e-8
        )

        return float(
            (similarity + 1.0)
            / 2.0
        )


    # ==================================================
    # DEVICE AFFINITY
    # ==================================================

    def _device_affinity(
        self,
        user_row: pd.Series,
        context_row: pd.Series
    ) -> float:
        """
        Device preference matching.
        """

        preferred = (
            user_row[
                "primary_device"
            ]
        )

        current = (
            context_row[
                "device_type"
            ]
        )

        if preferred == current:
            return 1.0

        return 0.50


    # ==================================================
    # TIME AFFINITY
    # ==================================================

    def _time_affinity(
        self,
        user_row: pd.Series,
        context_row: pd.Series
    ) -> float:
        """
        Preferred viewing hour.
        """

        preferred_hour = (
            user_row[
                "preferred_hour"
            ]
        )

        current_hour = (
            context_row[
                "hour_of_day"
            ]
        )

        distance = abs(
            preferred_hour
            - current_hour
        )

        distance = min(
            distance,
            24 - distance
        )

        return float(
            np.clip(
                1 - distance / 12,
                0,
                1
            )
        )


    # ==================================================
    # SEASONALITY AFFINITY
    # ==================================================

    def _seasonality_affinity(
        self,
        user_row: pd.Series,
        context_row: pd.Series
    ) -> float:
        """
        Seasonal content preferences.
        """

        preferred = (
            user_row[
                "preferred_season"
            ]
        )

        current = (
            context_row[
                "season"
            ]
        )

        return float(
            preferred == current
        )


    # ==================================================
    # MATURITY AFFINITY
    # ==================================================

    def _maturity_affinity(
        self,
        user_row: pd.Series,
        item_row: pd.Series
    ) -> float:
        """
        Age-rating compatibility.
        """

        user_tolerance = (
            user_row[
                "maturity_preference"
            ]
        )

        rating = (
            item_row[
                "maturity_rating"
            ]
        )

        mapping = {
            "G": 0.1,
            "PG": 0.3,
            "PG13": 0.5,
            "R": 0.8,
            "NC17": 1.0
        }

        item_level = mapping.get(
            rating,
            0.5
        )

        return float(
            1.0
            -
            abs(
                user_tolerance
                -
                item_level
            )
        )

    # ==================================================
    # CONTEXT AFFINITY
    # ==================================================

    def _context_affinity(
        self,
        context_row: pd.Series
    ) -> float:
        """
        Context attractiveness.

        Represents likelihood that
        recommendations work well
        under current conditions.
        """

        score = (
            context_row[
                "watch_intent_score"
            ] * 0.40
            +
            context_row[
                "attention_level"
            ] * 0.20
            +
            context_row[
                "network_quality_score"
            ] * 0.15
            +
            context_row[
                "context_quality_score"
            ] * 0.25
        )

        return float(
            np.clip(
                score,
                0,
                1
            )
        )

    # ==================================================
    # PERSONALIZATION SCORE
    # ==================================================

    def _personalization_score(
        self,
        user_row: pd.Series,
        item_row: pd.Series,
        context_row: pd.Series
    ) -> float:
        """
        Personalized utility score.
        """

        scores = [
            self._genre_affinity(
                user_row,
                item_row
            ),
            self._language_affinity(
                user_row,
                item_row
            ),
            self._maturity_affinity(
                user_row,
                item_row
            ),
            self._device_affinity(
                user_row,
                context_row
            ),
            self._time_affinity(
                user_row,
                context_row
            ),
        ]

        return float(
            np.mean(scores)
        )

    def calculate_affinity(
        self,
        user_row: pd.Series,
        item_row: pd.Series,
        context_row: pd.Series
    ) -> dict:
        """
        Production-grade affinity.
        """

        metadata_affinity = (
            super_metadata_affinity := (
                self._genre_affinity(
                    user_row,
                    item_row
                ) * 0.30
                +
                self._creator_affinity(
                    user_row,
                    item_row
                ) * 0.10
                +
                self._language_affinity(
                    user_row,
                    item_row
                ) * 0.10
                +
                self._popularity_affinity(
                    user_row,
                    item_row
                ) * 0.15
                +
                self._novelty_affinity(
                    user_row,
                    item_row
                ) * 0.10
                +
                self._quality_affinity(
                    item_row
                ) * 0.15
                +
                self._recency_affinity(
                    item_row
                ) * 0.10
            )
        )

        embedding_affinity = (
            self._user_item_embedding_affinity(
                user_row,
                item_row
            )
        )

        context_affinity = (
            self._context_affinity(
                context_row
            )
        )

        device_affinity = (
            self._device_affinity(
                user_row,
                context_row
            )
        )

        time_affinity = (
            self._time_affinity(
                user_row,
                context_row
            )
        )

        seasonality_affinity = (
            self._seasonality_affinity(
                user_row,
                context_row
            )
        )

        personalization_score = (
            self._personalization_score(
                user_row,
                item_row,
                context_row
            )
        )

        final_score = (
            metadata_affinity * 0.35
            +
            embedding_affinity * 0.25
            +
            context_affinity * 0.10
            +
            device_affinity * 0.05
            +
            time_affinity * 0.05
            +
            seasonality_affinity * 0.05
            +
            personalization_score * 0.15
        )

        final_score += self.rng.normal(
            0,
            0.02
        )

        final_score = np.clip(
            final_score,
            0,
            1
        )

        return {
            "affinity_score":
                float(final_score),

            "metadata_affinity":
                float(metadata_affinity),

            "embedding_affinity":
                float(embedding_affinity),

            "context_affinity":
                float(context_affinity),

            "device_affinity":
                float(device_affinity),

            "time_affinity":
                float(time_affinity),

            "seasonality_affinity":
                float(seasonality_affinity),

            "personalization_score":
                float(personalization_score)
        }

    # ==================================================
    # POSITION SENSITIVITY
    # ==================================================

    def _position_sensitivity(
        self,
        user_row: pd.Series
    ) -> float:
        """
        How strongly user behavior
        depends on rank position.
        """

        archetype = user_row[
            "archetype"
        ]

        mapping = {
            "Passive Viewer": 0.90,
            "Casual Browser": 0.80,
            "Trend Follower": 0.95,
            "Binge Watcher": 0.60,
            "Explorer": 0.30,
            "Critic": 0.50
        }

        return float(
            mapping.get(
                archetype,
                0.70
            )
        )    

    # ==================================================
    # NOVELTY SEEKING
    # ==================================================

    def _novelty_seeking(
        self,
        user_row: pd.Series
    ) -> float:
        """
        User desire for new content.
        """

        return float(
            user_row[
                "novelty_preference"
            ]
        )    

    # ==================================================
    # POPULARITY BIAS
    # ==================================================

    def _popularity_bias(
        self,
        user_row: pd.Series,
        item_row: pd.Series
    ) -> float:
        """
        Attraction toward trending items.
        """

        return (
            item_row[
                "popularity_score"
            ]
            *
            user_row[
                "popularity_preference"
            ]
        )

    # ==================================================
    # SERENDIPITY
    # ==================================================

    def _serendipity_score(
        self,
        user_row: pd.Series,
        item_row: pd.Series
    ) -> float:
        """
        Unexpected but relevant content.
        """

        genre_match = (
            self._genre_affinity(
                user_row,
                item_row
            )
        )

        novelty = (
            item_row[
                "freshness_score"
            ]
        )

        serendipity = (
            (1 - genre_match)
            * 0.5
            +
            novelty
            * 0.5
        )

        return float(
            np.clip(
                serendipity,
                0,
                1
            )
        )    

    # ==================================================
    # LONG TERM AFFINITY
    # ==================================================

    def _long_term_affinity(
        self,
        affinity_score: float,
        user_row: pd.Series
    ) -> float:
        """
        Stable preference component.
        """

        stability = (
            user_row[
                "engagement_stability"
            ]
        )

        score = (
            affinity_score
            * stability
        )

        return float(
            np.clip(
                score,
                0,
                1
            )
        )

    # ==================================================
    # SHORT TERM AFFINITY
    # ==================================================

    def _short_term_affinity(
        self,
        affinity_score: float,
        context_row: pd.Series
    ) -> float:
        """
        Session-driven preference.
        """

        score = (
            affinity_score
            *
            context_row[
                "watch_intent_score"
            ]
        )

        return float(
            np.clip(
                score,
                0,
                1
            )
        )

    # ==================================================
    # EXPLORATION AFFINITY
    # ==================================================

    def _exploration_affinity(
        self,
        serendipity_score: float,
        context_row: pd.Series
    ) -> float:

        score = (
            serendipity_score
            *
            (
                1.0
                if context_row[
                    "exploration_mode"
                ]
                else 0.40
            )
        )

        return float(
            np.clip(
                score,
                0,
                1
            )
        )


    # ==================================================
    # EXPLOITATION AFFINITY
    # ==================================================

    def _exploitation_affinity(
        self,
        affinity_score: float,
        context_row: pd.Series
    ) -> float:

        multiplier = (
            1.0
            if context_row[
                "binge_mode"
            ]
            else 0.60
        )

        return float(
            np.clip(
                affinity_score
                * multiplier,
                0,
                1
            )
        )

    # ==================================================
    # CLICK PROBABILITY
    # ==================================================

    def _click_probability(
        self,
        affinity_score: float,
        context_row: pd.Series
    ) -> float:

        probability = (
            affinity_score
            * 0.70
            +
            context_row[
                "homepage_bias"
            ]
            * 0.15
            +
            context_row[
                "watch_intent_score"
            ]
            * 0.15
        )

        probability += self.rng.normal(
            0,
            0.03
        )

        return float(
            np.clip(
                probability,
                0,
                1
            )
        )

    # ==================================================
    # WATCH PROBABILITY
    # ==================================================

    def _watch_probability(
        self,
        affinity_score: float,
        click_probability: float
    ) -> float:

        probability = (
            affinity_score
            * 0.50
            +
            click_probability
            * 0.50
        )

        return float(
            np.clip(
                probability,
                0,
                1
            )
        )

    # ==================================================
    # COMPLETION PROBABILITY
    # ==================================================

    def _completion_probability(
        self,
        watch_probability: float,
        item_row: pd.Series
    ) -> float:

        probability = (
            watch_probability
            * 0.70
            +
            item_row[
                "completion_rate"
            ]
            * 0.30
        )

        return float(
            np.clip(
                probability,
                0,
                1
            )
        )

    # ==================================================
    # SATISFACTION
    # ==================================================

    def _satisfaction_probability(
        self,
        affinity_score: float,
        completion_probability: float
    ) -> float:

        probability = (
            affinity_score
            * 0.50
            +
            completion_probability
            * 0.50
        )

        return float(
            np.clip(
                probability,
                0,
                1
            )
        )

    # ==================================================
    # RETENTION PROBABILITY
    # ==================================================

    def _retention_probability(
        self,
        satisfaction_probability: float,
        item_row: pd.Series
    ) -> float:

        probability = (
            satisfaction_probability
            * 0.70
            +
            item_row[
                "retention_impact"
            ]
            * 0.30
        )

        return float(
            np.clip(
                probability,
                0,
                1
            )
        )


    # ==================================================
    # CHURN REDUCTION
    # ==================================================

    def _churn_reduction_probability(
        self,
        retention_probability: float
    ) -> float:

        return float(
            np.clip(
                retention_probability
                * 0.95,
                0,
                1
            )
        )

    # ============================================================
    # AFFINITYENGINE PART 4 — PRODUCTION FINALIZATION LAYER
    # Add inside AffinityEngine class
    # ============================================================

    # ==================================================
    # AFFINITY EMBEDDING ID
    # ==================================================

    def _generate_affinity_embedding_id(
        self,
        affinity_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate unique affinity embedding ids.

        Returns
        -------
        np.ndarray
            Unique identifier for every user-item affinity row.
        """

        return np.arange(
            1,
            len(affinity_df) + 1,
            dtype=np.int64
        )


    # ==================================================
    # AFFINITY CLUSTER
    # ==================================================

    def _generate_affinity_cluster(
        self,
        affinity_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Segment affinity scores into clusters.

        Cluster Meaning
        ---------------
        0 : Very Low Affinity
        1 : Low Affinity
        2 : Medium Affinity
        3 : High Affinity
        4 : Very High Affinity
        """

        affinity_scores = (
            affinity_df[
                "affinity_score"
            ].values
        )

        clusters = np.digitize(
            affinity_scores,
            bins=[
                0.20,
                0.40,
                0.60,
                0.80
            ]
        )

        return clusters.astype(
            np.int32
        )


    # ==================================================
    # CANDIDATE GENERATION SCORE
    # ==================================================

    def _candidate_generation_score(
        self,
        affinity_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Retrieval-stage relevance score.

        Used by:
            Two-Tower
            ANN Retrieval
            Candidate Generation
        """

        score = (
            affinity_df[
                "affinity_score"
            ].values
            * 0.50
            +
            affinity_df[
                "watch_probability"
            ].values
            * 0.30
            +
            affinity_df[
                "click_probability"
            ].values
            * 0.20
        )

        return np.clip(
            score,
            0.0,
            1.0
        )


    # ==================================================
    # RANKING LABEL
    # ==================================================

    def _ranking_label(
        self,
        affinity_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Ground-truth ranking target.

        Used by:
            XGBoost Ranker
            LightGBM Ranker
            Deep Ranking Models
        """

        label = (
            affinity_df[
                "watch_probability"
            ].values
            * 0.40
            +
            affinity_df[
                "completion_probability"
            ].values
            * 0.30
            +
            affinity_df[
                "satisfaction_probability"
            ].values
            * 0.30
        )

        return np.clip(
            label,
            0.0,
            1.0
        )


    # ==================================================
    # BANDIT REWARD
    # ==================================================

    def _bandit_reward(
        self,
        affinity_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Reward signal for contextual bandits.
        """

        reward = (
            affinity_df[
                "retention_probability"
            ].values
            * 0.50
            +
            affinity_df[
                "satisfaction_probability"
            ].values
            * 0.50
        )

        return np.clip(
            reward,
            0.0,
            1.0
        )


    # ==================================================
    # POSITIVE INTERACTION LABEL
    # ==================================================

    def _positive_interaction_label(
        self,
        affinity_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Binary interaction target.

        Useful for:
            CTR prediction
            Retrieval training
        """

        probability = (
            affinity_df[
                "click_probability"
            ].values
        )

        return (
            probability >= 0.50
        ).astype(
            np.int8
        )


    # ==================================================
    # HIGH VALUE USER LABEL
    # ==================================================

    def _high_value_label(
        self,
        affinity_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Business-value training target.
        """

        score = (
            affinity_df[
                "retention_probability"
            ].values
        )

        return (
            score >= 0.75
        ).astype(
            np.int8
        )


    # ==================================================
    # USER ITEM SPARSE MATRIX
    # ==================================================

    def build_user_item_score_matrix(
        self,
        affinity_df: pd.DataFrame
    ):
        """
        Sparse affinity matrix.

        Used by:
            ALS
            Matrix Factorization
            LightGCN
            Graph Models
        """

        from scipy.sparse import csr_matrix

        users = (
            affinity_df[
                "user_id"
            ]
            .astype("category")
        )

        items = (
            affinity_df[
                "item_id"
            ]
            .astype("category")
        )

        matrix = csr_matrix(
            (
                affinity_df[
                    "affinity_score"
                ].values,
                (
                    users.cat.codes,
                    items.cat.codes
                )
            )
        )

        return matrix


    # ==================================================
    # FEATURE STORE EXPORT
    # ==================================================

    def build_affinity_feature_store(
        self,
        affinity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Offline + online feature store table.
        """

        features = affinity_df.copy()

        features[
            "event_timestamp"
        ] = pd.Timestamp.utcnow()

        return features


    # ==================================================
    # TWO TOWER DATASET
    # ==================================================

    def build_two_tower_dataset(
        self,
        affinity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Retrieval training dataset.
        """

        return affinity_df[
            [
                "user_id",
                "item_id",
                "candidate_generation_score",
                "positive_interaction_label"
            ]
        ].copy()


    # ==================================================
    # RANKING DATASET
    # ==================================================

    def build_ranking_dataset(
        self,
        affinity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Learning-to-rank dataset.
        """

        return affinity_df[
            [
                "user_id",
                "item_id",
                "ranking_label"
            ]
        ].copy()


    # ==================================================
    # BANDIT DATASET
    # ==================================================

    def build_bandit_dataset(
        self,
        affinity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Contextual bandit training dataset.
        """

        return affinity_df[
            [
                "user_id",
                "item_id",
                "bandit_reward"
            ]
        ].copy()


    # ==================================================
    # ONLINE SERVING FEATURES
    # ==================================================

    def build_serving_affinity_features(
        self,
        affinity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Real-time serving features.
        """

        columns = [
            "user_id",
            "item_id",
            "affinity_score",
            "candidate_generation_score",
            "ranking_label",
            "bandit_reward",
            "watch_probability",
            "click_probability",
            "retention_probability"
        ]

        return affinity_df[
            columns
        ].copy()


    # ==================================================
    # VALIDATION
    # ==================================================

    def validate_affinity_dataset(
        self,
        affinity_df: pd.DataFrame
    ) -> None:
        """
        Production-grade validation.
        """

        if affinity_df.empty:
            raise ValueError(
                "Affinity dataset is empty."
            )

        required_columns = [
            "user_id",
            "item_id",
            "affinity_score",
            "click_probability",
            "watch_probability",
            "completion_probability",
            "retention_probability"
        ]

        missing = [
            col
            for col in required_columns
            if col not in affinity_df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        probability_columns = [
            "affinity_score",
            "metadata_affinity",
            "embedding_affinity",
            "context_affinity",
            "click_probability",
            "watch_probability",
            "completion_probability",
            "satisfaction_probability",
            "retention_probability",
            "candidate_generation_score",
            "ranking_label",
            "bandit_reward"
        ]

        for col in probability_columns:

            if col not in affinity_df:
                continue

            values = (
                affinity_df[col]
                .values
            )

            if np.any(
                np.isnan(values)
            ):
                raise ValueError(
                    f"NaN detected in {col}"
                )

            if (
                values.min() < 0.0
                or
                values.max() > 1.0
            ):
                raise ValueError(
                    f"Invalid range in {col}"
                )

        if (
            affinity_df[
                ["user_id", "item_id"]
            ]
            .duplicated()
            .any()
        ):
            raise ValueError(
                "Duplicate user-item pairs detected."
            )


    # ==================================================
    # FINAL ENRICHMENT
    # ==================================================

    def enrich_affinity_dataset(
        self,
        affinity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add production-grade labels,
        scores and metadata.
        """

        affinity_df = affinity_df.copy()

        affinity_df[
            "affinity_embedding_id"
        ] = (
            self._generate_affinity_embedding_id(
                affinity_df
            )
        )

        affinity_df[
            "affinity_cluster"
        ] = (
            self._generate_affinity_cluster(
                affinity_df
            )
        )

        affinity_df[
            "candidate_generation_score"
        ] = (
            self._candidate_generation_score(
                affinity_df
            )
        )

        affinity_df[
            "ranking_label"
        ] = (
            self._ranking_label(
                affinity_df
            )
        )

        affinity_df[
            "bandit_reward"
        ] = (
            self._bandit_reward(
                affinity_df
            )
        )

        affinity_df[
            "positive_interaction_label"
        ] = (
            self._positive_interaction_label(
                affinity_df
            )
        )

        affinity_df[
            "high_value_label"
        ] = (
            self._high_value_label(
                affinity_df
            )
        )

        return affinity_df


    # ==================================================
    # COMPLETE GENERATION PIPELINE
    # ==================================================

    def generate_affinity_dataset(
        self,
        contexts: pd.DataFrame,
        candidates_per_user: int = 500
    ) -> pd.DataFrame:
        """
        Production-grade affinity generation.

        Pipeline
        --------
        User
            ↓
        Candidate Sampling
            ↓
        Affinity Computation
            ↓
        Behavioral Propensities
            ↓
        Labels
            ↓
        Validation

        Returns
        -------
        pd.DataFrame
        """

        rows = []

        for _, user in self.users.iterrows():

            candidate_indices = (
                self.sample_candidate_items(
                    user,
                    candidates_per_user
                )
            )

            candidate_items = (
                self.items.loc[
                    candidate_indices
                ]
            )

            context_row = (
                contexts.sample(
                    1,
                    random_state=int(
                        self.rng.integers(
                            0,
                            1_000_000
                        )
                    )
                )
                .iloc[0]
            )

            for _, item in (
                candidate_items.iterrows()
            ):

                affinity_dict = (
                    self.calculate_affinity(
                        user,
                        item,
                        context_row
                    )
                )

                rows.append(
                    {
                        "user_id":
                            user["user_id"],

                        "item_id":
                            item["item_id"],

                        "context_id":
                            context_row[
                                "context_id"
                            ],

                        **affinity_dict
                    }
                )

        affinity_df = pd.DataFrame(
            rows
        )

        affinity_df = (
            self.enrich_affinity_dataset(
                affinity_df
            )
        )

        self.validate_affinity_dataset(
            affinity_df
        )

        return affinity_df