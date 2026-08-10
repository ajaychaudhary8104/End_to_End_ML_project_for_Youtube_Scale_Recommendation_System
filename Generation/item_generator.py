from __future__ import annotations

import numpy as np
import pandas as pd


class ItemGenerator:
    """
    Production-grade content catalog generator.

    Generates:
        - content metadata
        - genres
        - languages
        - creators
        - release history

    Phase 2B Part 1
    """

    CONTENT_TYPES = [
        "Movie",
        "Series",
        "Short",
        "Documentary",
        "Live",
        "Podcast"
    ]

    GENRES = [
        "Action",
        "Adventure",
        "Comedy",
        "Drama",
        "Fantasy",
        "Horror",
        "Mystery",
        "Romance",
        "SciFi",
        "Thriller",
        "Documentary",
        "Animation",
        "Sports",
        "News",
        "Music"
    ]

    LANGUAGES = [
        "English",
        "Hindi",
        "German",
        "Japanese",
        "Portuguese",
        "Spanish",
        "French"
    ]

    COUNTRIES = [
        "USA",
        "India",
        "UK",
        "Germany",
        "Japan",
        "Brazil",
        "Canada",
        "Australia"
    ]

    MATURITY_RATINGS = [
        "Kids",
        "Teen",
        "Adult"
    ]

    STUDIOS = [
        "Studio_A",
        "Studio_B",
        "Studio_C",
        "Studio_D",
        "Studio_E",
        "Studio_F",
        "Studio_G"
    ]

    def __init__(
        self,
        foundation,
        item_embeddings: pd.DataFrame
    ):

        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng

        self.item_embeddings = item_embeddings

    # =====================================
    # TITLE GENERATION
    # =====================================

    def _generate_titles(
        self,
        n_items: int
    ) -> np.ndarray:

        return np.array(
            [
                f"Content_{i:07d}"
                for i in range(
                    1,
                    n_items + 1
                )
            ]
        )

    # =====================================
    # CONTENT TYPE
    # =====================================

    def _generate_content_type(
        self,
        n_items: int
    ) -> np.ndarray:

        return self.rng.choice(
            self.CONTENT_TYPES,
            size=n_items,
            p=[
                0.35,
                0.30,
                0.10,
                0.10,
                0.05,
                0.10
            ]
        )

    # =====================================
    # GENRES
    # =====================================

    def _sample_genres(
        self
    ) -> tuple:

        n = self.rng.integers(
            1,
            4
        )

        genres = self.rng.choice(
            self.GENRES,
            size=n,
            replace=False
        )

        primary = genres[0]

        return (
            primary,
            list(genres)
        )

    def _generate_genres(
        self,
        n_items: int
    ):

        primary = []
        subgenres = []

        for _ in range(n_items):

            p, s = self._sample_genres()

            primary.append(p)
            subgenres.append(s)

        return (
            np.array(primary),
            subgenres
        )

    # =====================================
    # LANGUAGE
    # =====================================

    def _generate_language(
        self,
        n_items: int
    ) -> np.ndarray:

        return self.rng.choice(
            self.LANGUAGES,
            size=n_items
        )

    # =====================================
    # COUNTRY
    # =====================================

    def _generate_country(
        self,
        n_items: int
    ) -> np.ndarray:

        return self.rng.choice(
            self.COUNTRIES,
            size=n_items
        )

    # =====================================
    # RELEASE DATES
    # =====================================

    def _generate_release_dates(
        self,
        n_items: int
    ) -> pd.Series:

        start = pd.Timestamp(
            "2010-01-01"
        )

        end = pd.Timestamp(
            self.config.end_date
        )

        days = (
            end - start
        ).days

        offsets = self.rng.integers(
            0,
            days,
            n_items
        )

        return pd.Series(
            start
            +
            pd.to_timedelta(
                offsets,
                unit="D"
            )
        )

    # =====================================
    # RUNTIME
    # =====================================

    def _generate_runtime(
        self,
        content_types
    ) -> np.ndarray:

        runtimes = []

        for ct in content_types:

            if ct == "Movie":
                runtimes.append(
                    self.rng.integers(
                        80,
                        180
                    )
                )

            elif ct == "Series":
                runtimes.append(
                    self.rng.integers(
                        20,
                        60
                    )
                )

            elif ct == "Short":
                runtimes.append(
                    self.rng.integers(
                        3,
                        20
                    )
                )

            else:
                runtimes.append(
                    self.rng.integers(
                        10,
                        120
                    )
                )

        return np.array(
            runtimes
        )

    # =====================================
    # MATURITY
    # =====================================

    def _generate_maturity(
        self,
        n_items: int
    ) -> np.ndarray:

        return self.rng.choice(
            self.MATURITY_RATINGS,
            size=n_items,
            p=[
                0.15,
                0.30,
                0.55
            ]
        )

    # =====================================
    # CREATORS
    # =====================================

    def _generate_creators(
        self,
        n_items: int
    ) -> np.ndarray:

        return np.array(
            [
                f"Creator_{i}"
                for i in self.rng.integers(
                    1,
                    5000,
                    n_items
                )
            ]
        )

    # =====================================
    # STUDIOS
    # =====================================

    def _generate_studios(
        self,
        n_items: int
    ) -> np.ndarray:

        return self.rng.choice(
            self.STUDIOS,
            size=n_items
        )

    # =====================================
    # MAIN GENERATION
    # =====================================

    def generate_base_items(
        self
    ) -> pd.DataFrame:

        n_items = self.config.n_items

        item_ids = np.arange(
            1,
            n_items + 1
        )

        content_type = (
            self._generate_content_type(
                n_items
            )
        )

        release_dates = (
            self._generate_release_dates(
                n_items
            )
        )

        primary_genre, sub_genres = (
            self._generate_genres(
                n_items
            )
        )

        items = pd.DataFrame(
            {
                "item_id":
                item_ids,

                "title":
                self._generate_titles(
                    n_items
                ),

                "content_type":
                content_type,

                "genre":
                primary_genre,

                "sub_genres":
                sub_genres,

                "language":
                self._generate_language(
                    n_items
                ),

                "country":
                self._generate_country(
                    n_items
                ),

                "release_date":
                release_dates,

                "release_year":
                release_dates.dt.year,

                "runtime_minutes":
                self._generate_runtime(
                    content_type
                ),

                "maturity_rating":
                self._generate_maturity(
                    n_items
                ),

                "creator":
                self._generate_creators(
                    n_items
                ),

                "studio":
                self._generate_studios(
                    n_items
                )
            }
        )

        return items

    # ==================================================
    # POPULARITY MODELING
    # ==================================================

    def _generate_popularity_score(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Global popularity score.

        Represents historical consumption volume.
        """

        score = self.rng.beta(
            a=2.5,
            b=5.0,
            size=len(items)
        )

        blockbuster_mask = (
            self.rng.random(len(items))
            < 0.05
        )

        score[blockbuster_mask] = np.clip(
            score[blockbuster_mask]
            + self.rng.uniform(
                0.30,
                0.60,
                blockbuster_mask.sum()
            ),
            0,
            1
        )

        return score


    # ==================================================
    # QUALITY MODELING
    # ==================================================

    def _generate_quality_score(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Intrinsic content quality.

        Long-term signal.
        """

        quality = self.rng.beta(
            a=4.0,
            b=2.0,
            size=len(items)
        )

        return np.clip(
            quality,
            0,
            1
        )


    # ==================================================
    # CRITIC SCORE
    # ==================================================

    def _generate_critic_score(
        self,
        quality_score: np.ndarray
    ) -> np.ndarray:
        """
        Critic score.

        Strongly correlated with quality.
        """

        score = (
            quality_score * 100
            +
            self.rng.normal(
                0,
                8,
                len(quality_score)
            )
        )

        return np.clip(
            score,
            0,
            100
        )


    # ==================================================
    # AUDIENCE SCORE
    # ==================================================

    def _generate_audience_score(
        self,
        popularity_score: np.ndarray,
        quality_score: np.ndarray
    ) -> np.ndarray:
        """
        Audience satisfaction.
        """

        score = (
            (
                popularity_score * 0.40
                +
                quality_score * 0.60
            ) * 100
        )

        score += self.rng.normal(
            0,
            10,
            len(score)
        )

        return np.clip(
            score,
            0,
            100
        )


    # ==================================================
    # VIRALITY SCORE
    # ==================================================

    def _generate_virality_score(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Viral burst potential.

        Most content is not viral.
        """

        virality = self.rng.beta(
            a=1.5,
            b=8.0,
            size=len(items)
        )

        viral_hits = (
            self.rng.random(len(items))
            < 0.02
        )

        virality[viral_hits] = self.rng.uniform(
            0.85,
            1.0,
            viral_hits.sum()
        )

        return virality


    # ==================================================
    # TREND SCORE
    # ==================================================

    def _generate_trend_score(
        self,
        items: pd.DataFrame,
        virality_score: np.ndarray
    ) -> np.ndarray:
        """
        Current trending signal.

        Strongly influenced by recency.
        """

        current_date = pd.Timestamp(
            self.config.end_date
        )

        age_days = (
            current_date
            - items["release_date"]
        ).dt.days.values

        freshness = np.exp(
            -age_days / 365
        )

        trend = (
            freshness * 0.70
            +
            virality_score * 0.30
        )

        trend += self.rng.normal(
            0,
            0.05,
            len(items)
        )

        return np.clip(
            trend,
            0,
            1
        )


    # ==================================================
    # EVERGREEN SCORE
    # ==================================================

    def _generate_evergreen_score(
        self,
        quality_score: np.ndarray
    ) -> np.ndarray:
        """
        Long-term catalog value.

        High-quality content ages well.
        """

        evergreen = (
            quality_score
            +
            self.rng.normal(
                0,
                0.05,
                len(quality_score)
            )
        )

        return np.clip(
            evergreen,
            0,
            1
        )


    # ==================================================
    # COMPLETION RATE
    # ==================================================

    def _generate_completion_rate(
        self,
        quality_score: np.ndarray
    ) -> np.ndarray:
        """
        Simulated completion rate.

        Useful for ranking labels.
        """

        completion = (
            0.35
            +
            quality_score * 0.60
        )

        completion += self.rng.normal(
            0,
            0.05,
            len(completion)
        )

        return np.clip(
            completion,
            0,
            1
        )


    # ==================================================
    # RETENTION IMPACT
    # ==================================================

    def _generate_retention_impact(
        self,
        quality_score: np.ndarray,
        evergreen_score: np.ndarray
    ) -> np.ndarray:
        """
        Contribution to retention.
        """

        retention = (
            quality_score * 0.60
            +
            evergreen_score * 0.40
        )

        return np.clip(
            retention,
            0,
            1
        )


    # ==================================================
    # BUSINESS VALUE SCORE
    # ==================================================

    def _generate_business_value_score(
        self,
        popularity_score: np.ndarray,
        quality_score: np.ndarray,
        retention_impact: np.ndarray
    ) -> np.ndarray:
        """
        Internal content value score.
        """

        score = (
            popularity_score * 0.40
            +
            quality_score * 0.30
            +
            retention_impact * 0.30
        )

        return np.clip(
            score,
            0,
            1
        )


    # ==================================================
    # ENRICH QUALITY FEATURES
    # ==================================================

    def enrich_quality_features(
        self,
        items: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Attach recommendation signals.
        """

        popularity_score = (
            self._generate_popularity_score(
                items
            )
        )

        quality_score = (
            self._generate_quality_score(
                items
            )
        )

        critic_score = (
            self._generate_critic_score(
                quality_score
            )
        )

        audience_score = (
            self._generate_audience_score(
                popularity_score,
                quality_score
            )
        )

        virality_score = (
            self._generate_virality_score(
                items
            )
        )

        trend_score = (
            self._generate_trend_score(
                items,
                virality_score
            )
        )

        evergreen_score = (
            self._generate_evergreen_score(
                quality_score
            )
        )

        completion_rate = (
            self._generate_completion_rate(
                quality_score
            )
        )

        retention_impact = (
            self._generate_retention_impact(
                quality_score,
                evergreen_score
            )
        )

        business_value_score = (
            self._generate_business_value_score(
                popularity_score,
                quality_score,
                retention_impact
            )
        )

        items["popularity_score"] = (
            popularity_score
        )

        items["quality_score"] = (
            quality_score
        )

        items["critic_score"] = (
            critic_score
        )

        items["audience_score"] = (
            audience_score
        )

        items["virality_score"] = (
            virality_score
        )

        items["trend_score"] = (
            trend_score
        )

        items["evergreen_score"] = (
            evergreen_score
        )

        items["completion_rate"] = (
            completion_rate
        )

        items["retention_impact"] = (
            retention_impact
        )

        items["business_value_score"] = (
            business_value_score
        )

        return items

    # ==================================================
    # CONTENT AGE
    # ==================================================

    def _generate_content_age_days(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Age of content relative to simulation end date.
        """

        current_date = pd.Timestamp(
            self.config.end_date
        )

        age_days = (
            current_date
            - items["release_date"]
        ).dt.days

        return age_days.astype(
            np.int32
        )


    # ==================================================
    # FRESHNESS SCORE
    # ==================================================

    def _generate_freshness_score(
        self,
        age_days: np.ndarray
    ) -> np.ndarray:
        """
        Exponential freshness decay.
        """

        freshness = np.exp(
            -age_days / 365
        )

        return np.clip(
            freshness,
            0,
            1
        )


    # ==================================================
    # LIFECYCLE STAGE
    # ==================================================

    def _generate_lifecycle_stage(
        self,
        age_days: np.ndarray
    ) -> np.ndarray:
        """
        Content lifecycle stage.
        """

        stage = np.empty(
            len(age_days),
            dtype=object
        )

        stage[age_days <= 30] = "launch"

        stage[
            (age_days > 30)
            &
            (age_days <= 180)
        ] = "growth"

        stage[
            (age_days > 180)
            &
            (age_days <= 730)
        ] = "mature"

        stage[
            age_days > 730
        ] = "decline"

        return stage


    # ==================================================
    # CATALOG STATUS
    # ==================================================

    def _generate_catalog_status(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Availability state.
        """

        return self.rng.choice(
            [
                "active",
                "licensed",
                "expiring",
                "retired"
            ],
            size=len(items),
            p=[
                0.78,
                0.15,
                0.05,
                0.02
            ]
        )


    # ==================================================
    # SEASONAL CONTENT
    # ==================================================

    def _generate_seasonal_flag(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Seasonal content indicator.
        """

        seasonal_genres = {
            "Sports",
            "Music",
            "News"
        }

        return items["genre"].isin(
            seasonal_genres
        ).values


    # ==================================================
    # LICENSING RISK
    # ==================================================

    def _generate_licensing_risk(
        self,
        catalog_status: np.ndarray
    ) -> np.ndarray:
        """
        Probability of leaving platform.
        """

        risk = np.zeros(
            len(catalog_status)
        )

        risk[
            catalog_status == "active"
        ] = self.rng.uniform(
            0.00,
            0.20,
            np.sum(
                catalog_status == "active"
            )
        )

        risk[
            catalog_status == "licensed"
        ] = self.rng.uniform(
            0.20,
            0.60,
            np.sum(
                catalog_status == "licensed"
            )
        )

        risk[
            catalog_status == "expiring"
        ] = self.rng.uniform(
            0.60,
            0.95,
            np.sum(
                catalog_status == "expiring"
            )
        )

        risk[
            catalog_status == "retired"
        ] = 1.0

        return risk


    # ==================================================
    # AVAILABILITY STATUS
    # ==================================================

    def _generate_availability_status(
        self,
        licensing_risk: np.ndarray
    ) -> np.ndarray:
        """
        Recommendation serving status.
        """

        status = np.full(
            len(licensing_risk),
            "available",
            dtype=object
        )

        status[
            licensing_risk > 0.75
        ] = "restricted"

        status[
            licensing_risk >= 1.0
        ] = "removed"

        return status


    # ==================================================
    # CATALOG SEGMENT
    # ==================================================

    def _generate_catalog_segment(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Strategic business categorization.
        """

        popularity = (
            items["popularity_score"]
            .values
        )

        quality = (
            items["quality_score"]
            .values
        )

        segment = np.full(
            len(items),
            "standard",
            dtype=object
        )

        segment[
            popularity > 0.80
        ] = "blockbuster"

        segment[
            quality > 0.90
        ] = "prestige"

        segment[
            (
                popularity < 0.20
            )
            &
            (
                quality > 0.75
            )
        ] = "hidden_gem"

        segment[
            items["trend_score"].values
            > 0.80
        ] = "trending"

        return segment


    # ==================================================
    # LAUNCH SCORE
    # ==================================================

    def _generate_launch_momentum(
        self,
        items: pd.DataFrame,
        freshness_score: np.ndarray
    ) -> np.ndarray:
        """
        Early-stage growth potential.
        """

        momentum = (
            freshness_score * 0.60
            +
            items["virality_score"].values * 0.40
        )

        return np.clip(
            momentum,
            0,
            1
        )


    # ==================================================
    # LONG TERM VALUE
    # ==================================================

    def _generate_long_term_value(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Long-term catalog utility.
        """

        score = (
            items["quality_score"].values
            * 0.50
            +
            items["evergreen_score"].values
            * 0.50
        )

        return np.clip(
            score,
            0,
            1
        )


    # ==================================================
    # LIFECYCLE ENRICHMENT
    # ==================================================

    def enrich_lifecycle_features(
        self,
        items: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add lifecycle features.
        """

        age_days = (
            self._generate_content_age_days(
                items
            )
        )

        freshness_score = (
            self._generate_freshness_score(
                age_days
            )
        )

        lifecycle_stage = (
            self._generate_lifecycle_stage(
                age_days
            )
        )

        catalog_status = (
            self._generate_catalog_status(
                items
            )
        )

        licensing_risk = (
            self._generate_licensing_risk(
                catalog_status
            )
        )

        items["content_age_days"] = (
            age_days
        )

        items["freshness_score"] = (
            freshness_score
        )

        items["lifecycle_stage"] = (
            lifecycle_stage
        )

        items["catalog_status"] = (
            catalog_status
        )

        items["seasonal_content_flag"] = (
            self._generate_seasonal_flag(
                items
            )
        )

        items["licensing_risk"] = (
            licensing_risk
        )

        items["availability_status"] = (
            self._generate_availability_status(
                licensing_risk
            )
        )

        items["catalog_segment"] = (
            self._generate_catalog_segment(
                items
            )
        )

        items["launch_momentum"] = (
            self._generate_launch_momentum(
                items,
                freshness_score
            )
        )

        items["long_term_value"] = (
            self._generate_long_term_value(
                items
            )
        )

        return items

    # ==================================================
    # EMBEDDING IDS
    # ==================================================

    def _generate_embedding_id(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        One embedding per item.
        """

        return items["item_id"].values


    # ==================================================
    # EMBEDDING CLUSTERS
    # ==================================================

    def _generate_embedding_cluster(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Latent semantic cluster.

        Used for:
            retrieval
            ANN partitioning
            catalog analysis
        """

        return self.rng.integers(
            0,
            100,
            len(items)
        )


    # ==================================================
    # GENRE EMBEDDINGS
    # ==================================================

    def _generate_genre_embedding(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Dense genre representation.

        Shape:
            [n_items, genre_dim]
        """

        embedding_dim = 32

        vectors = self.rng.normal(
            0,
            1,
            (
                len(items),
                embedding_dim
            )
        )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        return vectors / norms


    # ==================================================
    # CREATOR EMBEDDINGS
    # ==================================================

    def _generate_creator_embedding(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Creator latent vectors.
        """

        embedding_dim = 32

        vectors = self.rng.normal(
            0,
            1,
            (
                len(items),
                embedding_dim
            )
        )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        return vectors / norms


    # ==================================================
    # CONTENT EMBEDDINGS
    # ==================================================

    def _generate_content_embedding(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Main item embedding.

        Production systems often use:
            64
            128
            256
            512
        dimensions.
        """

        embedding_dim = getattr(
            self.config,
            "embedding_dim",
            128
        )

        vectors = self.rng.normal(
            0,
            1,
            (
                len(items),
                embedding_dim
            )
        )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        return vectors / norms


    # ==================================================
    # RETRIEVAL SCORE
    # ==================================================

    def _generate_retrieval_score(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Candidate-generation signal.
        """

        score = (
            items["popularity_score"].values
            * 0.35
            +
            items["trend_score"].values
            * 0.35
            +
            items["quality_score"].values
            * 0.30
        )

        return np.clip(
            score,
            0,
            1
        )


    # ==================================================
    # RANKING SCORE
    # ==================================================

    def _generate_ranking_score(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Primary ranking target.
        """

        score = (
            items["quality_score"].values
            * 0.40
            +
            items["completion_rate"].values
            * 0.25
            +
            items["audience_score"].values / 100 * 0.20
            +
            items["trend_score"].values
            * 0.15
        )

        return np.clip(
            score,
            0,
            1
        )


    # ==================================================
    # DIVERSITY SCORE
    # ==================================================

    def _generate_diversity_score(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Diversity promotion signal.
        """

        score = (
            1.0
            -
            items["popularity_score"].values
        )

        score += self.rng.normal(
            0,
            0.05,
            len(items)
        )

        return np.clip(
            score,
            0,
            1
        )


    # ==================================================
    # EXPLORATION SCORE
    # ==================================================

    def _generate_exploration_score(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Useful for bandits and re-ranking.
        """

        score = (
            items["freshness_score"].values
            * 0.50
            +
            items["diversity_score"].values
            * 0.50
        )

        return np.clip(
            score,
            0,
            1
        )


    # ==================================================
    # SERVING PRIORITY
    # ==================================================

    def _generate_serving_priority(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Online serving priority.
        """

        score = (
            items["retrieval_score"].values
            * 0.50
            +
            items["ranking_score"].values
            * 0.50
        )

        return np.clip(
            score,
            0,
            1
        )


    # ==================================================
    # FEATURE ENRICHMENT
    # ==================================================

    def enrich_recommendation_features(
        self,
        items: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Attach recommendation features.
        """

        items["item_embedding_id"] = (
            self._generate_embedding_id(
                items
            )
        )

        items["embedding_cluster"] = (
            self._generate_embedding_cluster(
                items
            )
        )

        items["retrieval_score"] = (
            self._generate_retrieval_score(
                items
            )
        )

        items["ranking_score"] = (
            self._generate_ranking_score(
                items
            )
        )

        items["diversity_score"] = (
            self._generate_diversity_score(
                items
            )
        )

        items["exploration_score"] = (
            self._generate_exploration_score(
                items
            )
        )

        items["serving_priority"] = (
            self._generate_serving_priority(
                items
            )
        )

        self.content_embedding_matrix = (
            self._generate_content_embedding(
                items
            )
        )

        self.genre_embedding_matrix = (
            self._generate_genre_embedding(
                items
            )
        )

        self.creator_embedding_matrix = (
            self._generate_creator_embedding(
                items
            )
        )

        return items

    # ==================================================
    # COLD START SIMULATION
    # ==================================================

    def _generate_cold_start_flag(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        New content not yet exposed to users.
        """

        return (
            items["content_age_days"].values
            <= 14
        )


    def _generate_cold_start_type(
        self,
        items: pd.DataFrame
    ) -> np.ndarray:
        """
        Cold-start category.
        """

        result = np.full(
            len(items),
            "none",
            dtype=object
        )

        new_mask = (
            items["content_age_days"].values
            <= 14
        )

        result[new_mask] = self.rng.choice(
            [
                "new_movie",
                "new_series",
                "new_catalog_entry"
            ],
            size=new_mask.sum(),
            p=[
                0.40,
                0.40,
                0.20
            ]
        )

        return result


    # ==================================================
    # TWO TOWER FEATURES
    # ==================================================

    def build_two_tower_features(
        self,
        items: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Retrieval training features.
        """

        return items[
            [
                "item_id",
                "genre",
                "language",
                "content_type",
                "quality_score",
                "popularity_score",
                "trend_score",
                "embedding_cluster"
            ]
        ].copy()


    # ==================================================
    # LIGHTGCN FEATURES
    # ==================================================

    def build_lightgcn_features(
        self,
        items: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Graph-training metadata.
        """

        return items[
            [
                "item_id",
                "genre",
                "creator",
                "embedding_cluster"
            ]
        ].copy()


    # ==================================================
    # FEATURE STORE EXPORT
    # ==================================================

    def build_item_feature_store(
        self,
        items: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Offline/online serving table.
        """

        features = items.copy()

        features[
            "event_timestamp"
        ] = pd.Timestamp.utcnow()

        return features


    # ==================================================
    # EMBEDDING EXPORT
    # ==================================================

    def export_item_embeddings(
        self
    ) -> pd.DataFrame:
        """
        Export item embeddings.
        """

        embeddings = pd.DataFrame(
            self.content_embedding_matrix
        )

        embeddings.insert(
            0,
            "item_id",
            np.arange(
                1,
                len(embeddings) + 1
            )
        )

        return embeddings


    # ==================================================
    # SERVING FEATURES
    # ==================================================

    def build_serving_features(
        self,
        items: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Online recommendation serving.
        """

        return items[
            [
                "item_id",
                "retrieval_score",
                "ranking_score",
                "serving_priority",
                "availability_status",
                "catalog_status",
                "trend_score"
            ]
        ].copy()


    # ==================================================
    # VALIDATION
    # ==================================================

    def validate_items(
        self,
        items: pd.DataFrame
    ) -> None:
        """
        Production validation.
        """

        if items.empty:
            raise ValueError(
                "Empty item dataset."
            )

        if (
            items["item_id"]
            .duplicated()
            .any()
        ):
            raise ValueError(
                "Duplicate item ids."
            )

        required = [
            "item_id",
            "genre",
            "language",
            "retrieval_score",
            "ranking_score"
        ]

        for col in required:

            if col not in items.columns:

                raise ValueError(
                    f"Missing column {col}"
                )

        score_columns = [
            "quality_score",
            "popularity_score",
            "trend_score",
            "retrieval_score",
            "ranking_score"
        ]

        for col in score_columns:

            if (
                items[col].min() < 0
                or
                items[col].max() > 1
            ):
                raise ValueError(
                    f"Invalid range in {col}"
                )


    # ==================================================
    # FINAL ENRICHMENT
    # ==================================================

    def enrich_final_features(
        self,
        items: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Final item enrichment.
        """

        items[
            "cold_start_item_flag"
        ] = (
            self._generate_cold_start_flag(
                items
            )
        )

        items[
            "cold_start_type"
        ] = (
            self._generate_cold_start_type(
                items
            )
        )

        return items


    # ==================================================
    # PRODUCTION GENERATE
    # ==================================================

    def generate(
        self
    ) -> pd.DataFrame:
        """
        Production-grade item generation.

        Pipeline
        --------

        Base Metadata
            ↓

        Popularity Layer
            ↓

        Lifecycle Layer
            ↓

        Recommendation Layer
            ↓

        Final Layer
            ↓

        Validation
        """

        items = (
            self.generate_base_items()
        )

        items = (
            self.enrich_quality_features(
                items
            )
        )

        items = (
            self.enrich_lifecycle_features(
                items
            )
        )

        items = (
            self.enrich_recommendation_features(
                items
            )
        )

        items = (
            self.enrich_final_features(
                items
            )
        )

        self.validate_items(
            items
        )

        return items