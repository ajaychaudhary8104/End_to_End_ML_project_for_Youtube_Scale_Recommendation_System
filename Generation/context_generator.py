from __future__ import annotations

import numpy as np
import pandas as pd


class ContextGenerator:
    """
    Production-grade context generation engine.

    Generates contextual features consumed by:

        SessionTimelineGenerator
        InteractionGenerator
        AffinityEngine
        PositionBiasEngine
        RankingGenerator
        BanditGenerator
        ReRankingGenerator
    """

    DEVICE_TYPES = [
        "Mobile",
        "Desktop",
        "Tablet",
        "SmartTV",
        "Console"
    ]

    NETWORK_TYPES = [
        "WiFi",
        "5G",
        "4G",
        "Ethernet"
    ]

    OS_TYPES = [
        "Android",
        "iOS",
        "Windows",
        "MacOS",
        "Linux",
        "TVOS"
    ]

    SURFACE_TYPES = [
        "homepage",
        "search",
        "details_page",
        "continue_watching",
        "trending",
        "recommended",
        "category_page"
    ]

    TRAFFIC_SOURCES = [
        "organic",
        "push_notification",
        "email",
        "advertisement",
        "social",
        "direct"
    ]

    SESSION_INTENTS = [
        "browse",
        "watch",
        "search",
        "continue",
        "explore"
    ]

    def __init__(
        self,
        foundation,
        users: pd.DataFrame
    ):
        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng
        self.users = users

    # =====================================================
    # TIMESTAMP GENERATION
    # =====================================================

    def _generate_timestamps(
        self,
        n_contexts: int
    ) -> pd.Series:

        start = pd.Timestamp(
            self.config.start_date
        )

        end = pd.Timestamp(
            self.config.end_date
        )

        total_seconds = int(
            (end - start).total_seconds()
        )

        offsets = self.rng.integers(
            0,
            total_seconds,
            size=n_contexts
        )

        return pd.Series(
            start +
            pd.to_timedelta(
                offsets,
                unit="s"
            )
        )

    # =====================================================
    # TEMPORAL FEATURES
    # =====================================================

    def _build_temporal_features(
        self,
        timestamps: pd.Series
    ) -> pd.DataFrame:

        df = pd.DataFrame()

        df["timestamp"] = timestamps

        df["date"] = timestamps.dt.date

        df["hour_of_day"] = (
            timestamps.dt.hour
        )

        df["day_of_week"] = (
            timestamps.dt.dayofweek
        )

        df["week_of_year"] = (
            timestamps.dt.isocalendar().week
        ).astype(int)

        df["month"] = (
            timestamps.dt.month
        )

        df["quarter"] = (
            timestamps.dt.quarter
        )

        return df

    # =====================================================
    # SEASON FEATURES
    # =====================================================

    def _generate_season(
        self,
        months: np.ndarray
    ) -> np.ndarray:

        season = np.empty(
            len(months),
            dtype=object
        )

        season[np.isin(
            months,
            [12, 1, 2]
        )] = "winter"

        season[np.isin(
            months,
            [3, 4, 5]
        )] = "spring"

        season[np.isin(
            months,
            [6, 7, 8]
        )] = "summer"

        season[np.isin(
            months,
            [9, 10, 11]
        )] = "autumn"

        return season

    # =====================================================
    # DEVICE CONTEXT
    # =====================================================

    def _generate_device_context(
        self,
        n_contexts: int
    ) -> pd.DataFrame:

        return pd.DataFrame(
            {
                "device_type":
                self.rng.choice(
                    self.DEVICE_TYPES,
                    size=n_contexts,
                    p=[
                        0.40,
                        0.20,
                        0.10,
                        0.25,
                        0.05
                    ]
                ),

                "network_type":
                self.rng.choice(
                    self.NETWORK_TYPES,
                    size=n_contexts,
                    p=[
                        0.55,
                        0.20,
                        0.15,
                        0.10
                    ]
                ),

                "os_type":
                self.rng.choice(
                    self.OS_TYPES,
                    size=n_contexts
                )
            }
        )

    # =====================================================
    # TRAFFIC CONTEXT
    # =====================================================

    def _generate_traffic_context(
        self,
        n_contexts: int
    ) -> pd.DataFrame:

        return pd.DataFrame(
            {
                "traffic_source":
                self.rng.choice(
                    self.TRAFFIC_SOURCES,
                    size=n_contexts,
                    p=[
                        0.45,
                        0.10,
                        0.08,
                        0.07,
                        0.10,
                        0.20
                    ]
                ),

                "campaign_id":
                self.rng.integers(
                    1,
                    1000,
                    size=n_contexts
                )
            }
        )

    # =====================================================
    # RECOMMENDATION CONTEXT
    # =====================================================

    def _generate_surface_context(
        self,
        n_contexts: int
    ) -> pd.DataFrame:

        return pd.DataFrame(
            {
                "surface_type":
                self.rng.choice(
                    self.SURFACE_TYPES,
                    size=n_contexts
                ),

                "recommendation_slot":
                self.rng.integers(
                    1,
                    51,
                    size=n_contexts
                ),

                "page_position":
                self.rng.integers(
                    1,
                    101,
                    size=n_contexts
                )
            }
        )

    # =====================================================
    # USER STATE
    # =====================================================

    def _generate_user_state(
        self,
        n_contexts: int
    ) -> pd.DataFrame:

        return pd.DataFrame(
            {
                "session_intent":
                self.rng.choice(
                    self.SESSION_INTENTS,
                    size=n_contexts
                ),

                "attention_level":
                self.rng.beta(
                    3,
                    2,
                    size=n_contexts
                ),

                "mood_proxy":
                self.rng.choice(
                    [
                        "relaxed",
                        "focused",
                        "casual",
                        "exploratory"
                    ],
                    size=n_contexts
                )
            }
        )

    # =====================================================
    # MAIN GENERATION
    # =====================================================

    def generate_base_contexts(
        self,
        n_contexts: int
    ) -> pd.DataFrame:

        timestamps = (
            self._generate_timestamps(
                n_contexts
            )
        )

        temporal = (
            self._build_temporal_features(
                timestamps
            )
        )

        temporal["season"] = (
            self._generate_season(
                temporal["month"].values
            )
        )

        device_context = (
            self._generate_device_context(
                n_contexts
            )
        )

        traffic_context = (
            self._generate_traffic_context(
                n_contexts
            )
        )

        surface_context = (
            self._generate_surface_context(
                n_contexts
            )
        )

        user_state = (
            self._generate_user_state(
                n_contexts
            )
        )

        contexts = pd.concat(
            [
                temporal,
                device_context,
                traffic_context,
                surface_context,
                user_state
            ],
            axis=1
        )

        contexts.insert(
            0,
            "context_id",
            np.arange(
                1,
                len(contexts) + 1
            )
        )

        return contexts

    # =====================================================
    # HOLIDAY MODELING
    # =====================================================

    def _generate_holiday_flags(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Simulate holidays.

        Major spikes in consumption.
        """

        holiday_months = {
            1,
            11,
            12
        }

        probability = np.where(
            contexts["month"].isin(
                holiday_months
            ),
            0.18,
            0.03
        )

        return (
            self.rng.random(
                len(contexts)
            )
            < probability
        )


    # =====================================================
    # SPECIAL EVENTS
    # =====================================================

    def _generate_special_event_flags(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Sports finals,
        concerts,
        product launches,
        major events.
        """

        return (
            self.rng.random(
                len(contexts)
            )
            < 0.05
        )


    # =====================================================
    # PROMOTION FLAGS
    # =====================================================

    def _generate_promotion_flags(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Marketing promotions.
        """

        return (
            self.rng.random(
                len(contexts)
            )
            < 0.10
        )


    # =====================================================
    # COUNTRY CONTEXT
    # =====================================================

    def _generate_country_context(
        self,
        n_contexts: int
    ) -> np.ndarray:

        countries = [
            "USA",
            "India",
            "UK",
            "Germany",
            "Japan",
            "Canada",
            "Australia",
            "Brazil"
        ]

        return self.rng.choice(
            countries,
            size=n_contexts,
            p=[
                0.30,
                0.20,
                0.10,
                0.08,
                0.07,
                0.08,
                0.07,
                0.10
            ]
        )


    # =====================================================
    # TIMEZONE CONTEXT
    # =====================================================

    def _generate_timezone_context(
        self,
        countries: np.ndarray
    ) -> np.ndarray:
        """
        Country -> timezone mapping.
        """

        mapping = {
            "USA": "America/New_York",
            "India": "Asia/Kolkata",
            "UK": "Europe/London",
            "Germany": "Europe/Berlin",
            "Japan": "Asia/Tokyo",
            "Canada": "America/Toronto",
            "Australia": "Australia/Sydney",
            "Brazil": "America/Sao_Paulo"
        }

        return np.array(
            [
                mapping[c]
                for c in countries
            ]
        )


    # =====================================================
    # REAL-TIME DEMAND INDEX
    # =====================================================

    def _generate_real_time_demand(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Current platform demand.
        """

        hour = contexts[
            "hour_of_day"
        ].values

        demand = (
            np.sin(
                (hour / 24)
                * np.pi
            )
            * 0.5
            + 0.5
        )

        demand += self.rng.normal(
            0,
            0.05,
            len(hour)
        )

        return np.clip(
            demand,
            0,
            1
        )


    # =====================================================
    # STREAMING LOAD
    # =====================================================

    def _generate_streaming_load(
        self,
        demand_index: np.ndarray
    ) -> np.ndarray:
        """
        Platform load.
        """

        load = (
            demand_index
            +
            self.rng.normal(
                0,
                0.05,
                len(demand_index)
            )
        )

        return np.clip(
            load,
            0,
            1
        )


    # =====================================================
    # NETWORK QUALITY
    # =====================================================

    def _generate_network_quality(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Device/network quality signal.
        """

        network = contexts[
            "network_type"
        ].values

        quality = np.zeros(
            len(network)
        )

        quality[
            network == "Ethernet"
        ] = self.rng.uniform(
            0.90,
            1.00,
            np.sum(
                network == "Ethernet"
            )
        )

        quality[
            network == "WiFi"
        ] = self.rng.uniform(
            0.70,
            1.00,
            np.sum(
                network == "WiFi"
            )
        )

        quality[
            network == "5G"
        ] = self.rng.uniform(
            0.75,
            0.95,
            np.sum(
                network == "5G"
            )
        )

        quality[
            network == "4G"
        ] = self.rng.uniform(
            0.50,
            0.85,
            np.sum(
                network == "4G"
            )
        )

        return quality


    # =====================================================
    # WEATHER CONTEXT
    # =====================================================

    def _generate_weather_context(
        self,
        n_contexts: int
    ) -> np.ndarray:
        """
        Simplified weather states.
        """

        return self.rng.choice(
            [
                "sunny",
                "cloudy",
                "rainy",
                "stormy",
                "snowy"
            ],
            size=n_contexts,
            p=[
                0.35,
                0.25,
                0.20,
                0.05,
                0.15
            ]
        )


    # =====================================================
    # EVENT CONTEXT
    # =====================================================

    def _generate_event_context(
        self,
        special_event_flags: np.ndarray
    ) -> np.ndarray:
        """
        Event category.
        """

        event_types = np.array(
            [
                "none"
            ] * len(
                special_event_flags
            ),
            dtype=object
        )

        idx = np.where(
            special_event_flags
        )[0]

        if len(idx) > 0:

            event_types[idx] = (
                self.rng.choice(
                    [
                        "sports",
                        "concert",
                        "festival",
                        "movie_launch",
                        "award_show"
                    ],
                    size=len(idx)
                )
            )

        return event_types


    # =====================================================
    # AD CAMPAIGN CONTEXT
    # =====================================================

    def _generate_ad_campaign_context(
        self,
        promotion_flags: np.ndarray
    ) -> np.ndarray:
        """
        Marketing channel.
        """

        campaign = np.array(
            [
                "none"
            ] * len(
                promotion_flags
            ),
            dtype=object
        )

        idx = np.where(
            promotion_flags
        )[0]

        if len(idx) > 0:

            campaign[idx] = (
                self.rng.choice(
                    [
                        "email",
                        "push",
                        "social",
                        "paid_search",
                        "affiliate"
                    ],
                    size=len(idx)
                )
            )

        return campaign


    # =====================================================
    # CONTEXT ENRICHMENT
    # =====================================================

    def enrich_business_context(
        self,
        contexts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add production-grade
        business context.
        """

        holiday_flags = (
            self._generate_holiday_flags(
                contexts
            )
        )

        special_event_flags = (
            self._generate_special_event_flags(
                contexts
            )
        )

        promotion_flags = (
            self._generate_promotion_flags(
                contexts
            )
        )

        countries = (
            self._generate_country_context(
                len(contexts)
            )
        )

        contexts[
            "holiday_flag"
        ] = holiday_flags

        contexts[
            "special_event_flag"
        ] = special_event_flags

        contexts[
            "promotion_flag"
        ] = promotion_flags

        contexts[
            "country_context"
        ] = countries

        contexts[
            "timezone_context"
        ] = (
            self._generate_timezone_context(
                countries
            )
        )

        demand_index = (
            self._generate_real_time_demand(
                contexts
            )
        )

        contexts[
            "real_time_demand_index"
        ] = demand_index

        contexts[
            "streaming_load_index"
        ] = (
            self._generate_streaming_load(
                demand_index
            )
        )

        contexts[
            "network_quality_score"
        ] = (
            self._generate_network_quality(
                contexts
            )
        )

        contexts[
            "weather_context"
        ] = (
            self._generate_weather_context(
                len(contexts)
            )
        )

        contexts[
            "event_context"
        ] = (
            self._generate_event_context(
                special_event_flags
            )
        )

        contexts[
            "ad_campaign_context"
        ] = (
            self._generate_ad_campaign_context(
                promotion_flags
            )
        )

        return contexts

    # =====================================================
    # HOMEPAGE BIAS
    # =====================================================

    def _generate_homepage_bias(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Homepage users consume more recommendations.
        """

        surface = contexts[
            "surface_type"
        ].values

        bias = np.full(
            len(surface),
            0.3
        )

        bias[
            surface == "homepage"
        ] = self.rng.uniform(
            0.7,
            1.0,
            np.sum(
                surface == "homepage"
            )
        )

        return bias


    # =====================================================
    # SEARCH BIAS
    # =====================================================

    def _generate_search_bias(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Search sessions have lower recommendation
        dependency.
        """

        surface = contexts[
            "surface_type"
        ].values

        score = np.full(
            len(surface),
            0.2
        )

        score[
            surface == "search"
        ] = self.rng.uniform(
            0.7,
            1.0,
            np.sum(
                surface == "search"
            )
        )

        return score


    # =====================================================
    # WATCH INTENT
    # =====================================================

    def _generate_watch_intent_score(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Probability user intends to watch content.
        """

        score = (
            contexts[
                "attention_level"
            ].values
            * 0.6
            +
            contexts[
                "real_time_demand_index"
            ].values
            * 0.4
        )

        score += self.rng.normal(
            0,
            0.05,
            len(score)
        )

        return np.clip(
            score,
            0,
            1
        )


    # =====================================================
    # PURCHASE INTENT
    # =====================================================

    def _generate_purchase_intent_score(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Useful for premium upgrades
        and commerce recommendations.
        """

        score = self.rng.beta(
            2,
            8,
            len(contexts)
        )

        return np.clip(
            score,
            0,
            1
        )


    # =====================================================
    # SESSION DEPTH
    # =====================================================

    def _generate_session_depth(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Number of items likely explored.
        """

        depth = np.round(
            self.rng.lognormal(
                mean=2.0,
                sigma=0.7,
                size=len(contexts)
            )
        )

        return np.clip(
            depth,
            1,
            100
        ).astype(
            np.int32
        )


    # =====================================================
    # EXPLORATION MODE
    # =====================================================

    def _generate_exploration_mode(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        User browsing for discovery.
        """

        return (
            self.rng.random(
                len(contexts)
            )
            < 0.30
        )


    # =====================================================
    # BINGE MODE
    # =====================================================

    def _generate_binge_mode(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Long-session behavior.
        """

        return (
            contexts[
                "session_intent"
            ].values
            == "continue"
        ) | (
            self.rng.random(
                len(contexts)
            )
            < 0.15
        )


    # =====================================================
    # MULTI SCREEN
    # =====================================================

    def _generate_multi_screen_flag(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Device switching behavior.
        """

        return (
            self.rng.random(
                len(contexts)
            )
            < 0.12
        )


    # =====================================================
    # RECOMMENDATION COMPETITION
    # =====================================================

    def _generate_recommendation_competition(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Number of competing items visible.
        """

        competition = (
            contexts[
                "page_position"
            ].values
            +
            self.rng.normal(
                0,
                5,
                len(contexts)
            )
        )

        return np.clip(
            competition,
            1,
            100
        )


    # =====================================================
    # CONTENT SUPPLY PRESSURE
    # =====================================================

    def _generate_content_supply_pressure(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Catalog pressure.

        Higher means more content competing.
        """

        pressure = self.rng.beta(
            5,
            2,
            len(contexts)
        )

        return np.clip(
            pressure,
            0,
            1
        )


    # =====================================================
    # RANKING PRESSURE
    # =====================================================

    def _generate_ranking_pressure(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Difficulty of obtaining engagement.
        """

        score = (
            contexts[
                "content_supply_pressure"
            ].values
            * 0.6
            +
            contexts[
                "recommendation_competition"
            ].values
            / 100
            * 0.4
        )

        return np.clip(
            score,
            0,
            1
        )


    # =====================================================
    # CONTEXT RECOMMENDATION FEATURES
    # =====================================================

    def enrich_recommendation_context(
        self,
        contexts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Recommendation-system context layer.
        """

        contexts[
            "homepage_bias"
        ] = (
            self._generate_homepage_bias(
                contexts
            )
        )

        contexts[
            "search_bias"
        ] = (
            self._generate_search_bias(
                contexts
            )
        )

        contexts[
            "watch_intent_score"
        ] = (
            self._generate_watch_intent_score(
                contexts
            )
        )

        contexts[
            "purchase_intent_score"
        ] = (
            self._generate_purchase_intent_score(
                contexts
            )
        )

        contexts[
            "session_depth"
        ] = (
            self._generate_session_depth(
                contexts
            )
        )

        contexts[
            "exploration_mode"
        ] = (
            self._generate_exploration_mode(
                contexts
            )
        )

        contexts[
            "binge_mode"
        ] = (
            self._generate_binge_mode(
                contexts
            )
        )

        contexts[
            "multi_screen_flag"
        ] = (
            self._generate_multi_screen_flag(
                contexts
            )
        )

        contexts[
            "recommendation_competition"
        ] = (
            self._generate_recommendation_competition(
                contexts
            )
        )

        contexts[
            "content_supply_pressure"
        ] = (
            self._generate_content_supply_pressure(
                contexts
            )
        )

        contexts[
            "ranking_pressure_score"
        ] = (
            self._generate_ranking_pressure(
                contexts
            )
        )

        return contexts

    # =====================================================
    # CONTEXT EMBEDDING IDS
    # =====================================================

    def _generate_context_embedding_id(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Unique embedding identifier.

        Used for:
            context tower
            ANN retrieval
            serving cache
        """

        return contexts[
            "context_id"
        ].values


    # =====================================================
    # CONTEXT CLUSTER
    # =====================================================

    def _generate_context_cluster(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Context segmentation.

        Similar contexts
        belong to same cluster.
        """

        return self.rng.integers(
            0,
            50,
            size=len(contexts)
        )


    # =====================================================
    # CONTEXT QUALITY SCORE
    # =====================================================

    def _generate_context_quality_score(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Overall contextual richness.

        Useful for:
            ranking
            experimentation
            analysis
        """

        score = (
            contexts[
                "attention_level"
            ].values
            * 0.30
            +
            contexts[
                "network_quality_score"
            ].values
            * 0.25
            +
            contexts[
                "watch_intent_score"
            ].values
            * 0.25
            +
            (
                1.0 -
                contexts[
                    "ranking_pressure_score"
                ].values
            )
            * 0.20
        )

        score += self.rng.normal(
            0,
            0.03,
            len(score)
        )

        return np.clip(
            score,
            0,
            1
        )


    # =====================================================
    # CONTEXT EMBEDDINGS
    # =====================================================

    def _generate_context_embeddings(
        self,
        contexts: pd.DataFrame
    ) -> np.ndarray:
        """
        Dense context vectors.

        Used by:
            Two-Tower
            Deep Retrieval
            Ranking Models
        """

        embedding_dim = getattr(
            self.config,
            "embedding_dim",
            128
        )

        embeddings = self.rng.normal(
            0,
            1,
            (
                len(contexts),
                embedding_dim
            )
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        return embeddings / norms


    # =====================================================
    # FEATURE STORE EXPORT
    # =====================================================

    def build_context_feature_store(
        self,
        contexts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Offline + online serving table.
        """

        features = contexts.copy()

        features[
            "event_timestamp"
        ] = pd.Timestamp.utcnow()

        return features


    # =====================================================
    # SERVING FEATURES
    # =====================================================

    def build_serving_context_features(
        self,
        contexts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Real-time serving features.
        """

        return contexts[
            [
                "context_id",
                "watch_intent_score",
                "homepage_bias",
                "search_bias",
                "ranking_pressure_score",
                "context_quality_score",
                "real_time_demand_index",
                "network_quality_score"
            ]
        ].copy()


    # =====================================================
    # EMBEDDING EXPORT
    # =====================================================

    def export_context_embeddings(
        self
    ) -> pd.DataFrame:
        """
        Export context vectors.
        """

        embeddings = pd.DataFrame(
            self.context_embedding_matrix
        )

        embeddings.insert(
            0,
            "context_id",
            np.arange(
                1,
                len(embeddings) + 1
            )
        )

        return embeddings


    # =====================================================
    # VALIDATION
    # =====================================================

    def validate_contexts(
        self,
        contexts: pd.DataFrame
    ) -> None:
        """
        Production validation.
        """

        if contexts.empty:
            raise ValueError(
                "Context dataset is empty."
            )

        if (
            contexts["context_id"]
            .duplicated()
            .any()
        ):
            raise ValueError(
                "Duplicate context ids."
            )

        required_columns = [
            "context_id",
            "watch_intent_score",
            "ranking_pressure_score",
            "context_quality_score"
        ]

        for col in required_columns:

            if col not in contexts.columns:

                raise ValueError(
                    f"Missing column: {col}"
                )

        score_columns = [
            "watch_intent_score",
            "purchase_intent_score",
            "network_quality_score",
            "ranking_pressure_score",
            "context_quality_score"
        ]

        for col in score_columns:

            if (
                contexts[col].min() < 0
                or
                contexts[col].max() > 1
            ):
                raise ValueError(
                    f"Invalid values in {col}"
                )


    # =====================================================
    # FINAL ENRICHMENT
    # =====================================================

    def enrich_final_context_features(
        self,
        contexts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Final production enrichment.
        """

        contexts[
            "context_embedding_id"
        ] = (
            self._generate_context_embedding_id(
                contexts
            )
        )

        contexts[
            "context_cluster"
        ] = (
            self._generate_context_cluster(
                contexts
            )
        )

        contexts[
            "context_quality_score"
        ] = (
            self._generate_context_quality_score(
                contexts
            )
        )

        self.context_embedding_matrix = (
            self._generate_context_embeddings(
                contexts
            )
        )

        return contexts


    # =====================================================
    # PRODUCTION GENERATE
    # =====================================================

    def generate(
        self,
        n_contexts: int
    ) -> pd.DataFrame:
        """
        Production-grade context generation.

        Pipeline
        --------

        Base Context
            ↓

        Business Layer
            ↓

        Recommendation Layer
            ↓

        Embedding Layer
            ↓

        Validation
        """

        contexts = (
            self.generate_base_contexts(
                n_contexts
            )
        )

        contexts = (
            self.enrich_business_context(
                contexts
            )
        )

        contexts = (
            self.enrich_recommendation_context(
                contexts
            )
        )

        contexts = (
            self.enrich_final_context_features(
                contexts
            )
        )

        self.validate_contexts(
            contexts
        )

        return contexts