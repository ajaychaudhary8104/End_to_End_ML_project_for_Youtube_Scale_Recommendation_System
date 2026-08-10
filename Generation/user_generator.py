from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class UserGenerator:
    """
    Production-grade user generator.

    Responsibilities
    ----------------
    Generate realistic recommendation-system users
    for Netflix / YouTube / Amazon style systems.

    Generates:
        - demographics
        - geography
        - registration history
        - household structure
        - subscription plans

    Phase 2A (Part 1)
    -----------------
    Demographics
    Geography
    Registration
    Subscription
    Household

    Later Parts
    -----------
    Part 2:
        Behavioral Modeling

    Part 3:
        Preferences & Affinities

    Part 4:
        Final Assembly & Validation
    """

    # --------------------------------------------------
    # INIT
    # --------------------------------------------------

    def __init__(
        self,
        foundation,
        personas: pd.DataFrame,
        archetypes: pd.DataFrame,
        user_embeddings: pd.DataFrame
    ):

        self.foundation = foundation
        self.config = foundation.config
        self.rng = foundation.rng

        self.personas = personas
        self.archetypes = archetypes
        self.user_embeddings = user_embeddings

    # ==================================================
    # DEMOGRAPHICS
    # ==================================================

    def _generate_age(self) -> np.ndarray:
        """
        Realistic streaming-platform age distribution.
        """

        age_buckets = [
            (13, 17),
            (18, 24),
            (25, 34),
            (35, 44),
            (45, 54),
            (55, 70)
        ]

        probs = np.array(
            [
                0.08,
                0.20,
                0.28,
                0.20,
                0.15,
                0.09
            ]
        )

        chosen = self.rng.choice(
            len(age_buckets),
            size=self.config.n_users,
            p=probs
        )

        ages = np.zeros(
            self.config.n_users,
            dtype=np.int16
        )

        for idx, (low, high) in enumerate(age_buckets):

            mask = chosen == idx

            ages[mask] = self.rng.integers(
                low,
                high + 1,
                mask.sum()
            )

        return ages

    def _generate_age_group(
        self,
        ages: np.ndarray
    ) -> np.ndarray:

        bins = [
            0,
            18,
            25,
            35,
            45,
            55,
            100
        ]

        labels = [
            "teen",
            "young_adult",
            "adult",
            "mid_age",
            "mature",
            "senior"
        ]

        return pd.cut(
            ages,
            bins=bins,
            labels=labels,
            include_lowest=True
        ).astype(str)

    def _generate_gender(self) -> np.ndarray:

        return self.rng.choice(
            [
                "male",
                "female",
                "other"
            ],
            size=self.config.n_users,
            p=[
                0.49,
                0.49,
                0.02
            ]
        )

    # ==================================================
    # GEOGRAPHY
    # ==================================================

    def _generate_geography(
        self
    ) -> pd.DataFrame:
        """
        Generate country / region / language.
        """

        countries = pd.DataFrame(
            {
                "country": [
                    "USA",
                    "India",
                    "UK",
                    "Canada",
                    "Germany",
                    "Brazil",
                    "Japan",
                    "Australia"
                ],

                "language": [
                    "English",
                    "Hindi",
                    "English",
                    "English",
                    "German",
                    "Portuguese",
                    "Japanese",
                    "English"
                ],

                "timezone": [
                    "US/Eastern",
                    "Asia/Kolkata",
                    "Europe/London",
                    "America/Toronto",
                    "Europe/Berlin",
                    "America/Sao_Paulo",
                    "Asia/Tokyo",
                    "Australia/Sydney"
                ],

                "prob": [
                    0.30,
                    0.25,
                    0.10,
                    0.08,
                    0.08,
                    0.07,
                    0.07,
                    0.05
                ]
            }
        )

        sampled = countries.sample(
            n=self.config.n_users,
            replace=True,
            weights="prob",
            random_state=self.config.random_state
        )

        sampled = sampled.reset_index(
            drop=True
        )

        sampled["region"] = self.rng.choice(
            [
                "Urban",
                "Suburban",
                "Rural"
            ],
            size=self.config.n_users,
            p=[
                0.60,
                0.30,
                0.10
            ]
        )

        return sampled

    # ==================================================
    # REGISTRATION
    # ==================================================

    def _generate_registration_dates(
        self
    ) -> pd.Series:

        start = pd.Timestamp(
            self.config.start_date
        )

        end = pd.Timestamp(
            self.config.end_date
        )

        total_days = (
            end - start
        ).days

        offsets = self.rng.integers(
            0,
            total_days,
            self.config.n_users
        )

        return pd.Series(
            start +
            pd.to_timedelta(
                offsets,
                unit="D"
            )
        )

    def _calculate_tenure(
        self,
        registration_dates
    ) -> np.ndarray:

        end = pd.Timestamp(
            self.config.end_date
        )

        tenure = (
            end - registration_dates
        ).dt.days

        return tenure.astype(
            np.int32
        )

    # ==================================================
    # HOUSEHOLDS
    # ==================================================

    def _generate_households(
        self
    ) -> pd.DataFrame:

        household_sizes = self.rng.choice(
            [
                1,
                2,
                3,
                4,
                5
            ],
            size=self.config.n_users,
            p=[
                0.30,
                0.28,
                0.20,
                0.15,
                0.07
            ]
        )

        household_ids = []

        counter = 1

        for size in household_sizes:

            household_ids.append(
                f"H{counter:08d}"
            )

            counter += 1

        return pd.DataFrame(
            {
                "household_id":
                household_ids,

                "household_size":
                household_sizes
            }
        )

    # ==================================================
    # SUBSCRIPTIONS
    # ==================================================

    def _generate_subscription_plan(
        self
    ) -> pd.DataFrame:

        plans = np.array(
            [
                "free",
                "basic",
                "standard",
                "premium",
                "family"
            ]
        )

        probs = np.array(
            [
                0.15,
                0.25,
                0.30,
                0.20,
                0.10
            ]
        )

        plan = self.rng.choice(
            plans,
            size=self.config.n_users,
            p=probs
        )

        prices = {
            "free": 0.0,
            "basic": 7.99,
            "standard": 12.99,
            "premium": 19.99,
            "family": 24.99
        }

        monthly_price = np.array(
            [
                prices[p]
                for p in plan
            ]
        )

        return pd.DataFrame(
            {
                "subscription_plan":
                plan,

                "monthly_price":
                monthly_price,

                "is_premium":
                np.isin(
                    plan,
                    [
                        "premium",
                        "family"
                    ]
                )
            }
        )

    # ==================================================
    # PERSONA / ARCHETYPE
    # ==================================================

    def _assign_personas(
        self
    ) -> np.ndarray:

        return self.rng.choice(
            self.personas[
                "persona_id"
            ].values,
            size=self.config.n_users
        )

    def _assign_archetypes(
        self
    ) -> np.ndarray:

        return self.rng.choice(
            self.archetypes[
                "archetype_id"
            ].values,
            size=self.config.n_users
        )

    # ==================================================
    # MAIN GENERATION
    # ==================================================

    def generate_base_users(
        self
    ) -> pd.DataFrame:
        """
        Generate user foundation table.

        Returns
        -------
        pd.DataFrame
        """

        n_users = self.config.n_users

        user_ids = np.arange(
            1,
            n_users + 1
        )

        ages = self._generate_age()

        users = pd.DataFrame(
            {
                "user_id":
                user_ids,

                "persona_id":
                self._assign_personas(),

                "archetype_id":
                self._assign_archetypes(),

                "age":
                ages,

                "age_group":
                self._generate_age_group(
                    ages
                ),

                "gender":
                self._generate_gender()
            }
        )

        geo = self._generate_geography()

        registration = (
            self._generate_registration_dates()
        )

        households = (
            self._generate_households()
        )

        subscriptions = (
            self._generate_subscription_plan()
        )

        users = pd.concat(
            [
                users,
                geo,
                households,
                subscriptions
            ],
            axis=1
        )

        users[
            "registration_date"
        ] = registration

        users[
            "tenure_days"
        ] = self._calculate_tenure(
            registration
        )

        users[
            "embedding_id"
        ] = users[
            "user_id"
        ]

        return users

    # ==================================================
    # VALIDATION
    # ==================================================

    def validate(
        self,
        users: pd.DataFrame
    ) -> None:
        """
        Basic quality checks.
        """

        if users.empty:
            raise ValueError(
                "users dataframe empty"
            )

        if users[
            "user_id"
        ].duplicated().any():

            raise ValueError(
                "duplicate user ids found"
            )

        if users[
            "age"
        ].isna().any():

            raise ValueError(
                "missing age values"
            )

        if users[
            "subscription_plan"
        ].isna().any():

            raise ValueError(
                "missing plans"
            )

    # ==================================================
    # PUBLIC API
    # ==================================================

    def generate(
        self
    ) -> pd.DataFrame:
        """
        Main generation entrypoint.
        """

        users = (
            self.generate_base_users()
        )

        self.validate(
            users
        )

        return users

    # ==================================================
    # BEHAVIORAL MODELING
    # ==================================================

    def _build_persona_lookup(self) -> pd.DataFrame:
        """
        Fast lookup table for persona features.
        """

        return (
            self.personas
            .set_index("persona_id")
            .copy()
        )


    def _build_archetype_lookup(self) -> pd.DataFrame:
        """
        Fast lookup table for archetype features.
        """

        return (
            self.archetypes
            .set_index("archetype_id")
            .copy()
        )


    def _generate_engagement_score(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate normalized engagement score.

        Range:
            0.0 -> 1.0
        """

        persona_lookup = self._build_persona_lookup()

        base = (
            users["persona_id"]
            .map(
                persona_lookup["engagement_score"]
            )
            .values
        )

        tenure_boost = np.clip(
            users["tenure_days"].values / 1000,
            0,
            1
        ) * 0.15

        premium_boost = (
            users["is_premium"]
            .astype(int)
            .values
            * 0.08
        )

        noise = self.rng.normal(
            0,
            0.05,
            len(users)
        )

        score = (
            base
            + tenure_boost
            + premium_boost
            + noise
        )

        return np.clip(
            score,
            0.01,
            1.0
        )


    def _generate_churn_risk(
        self,
        users: pd.DataFrame,
        engagement_score: np.ndarray
    ) -> np.ndarray:
        """
        Probability of churn.

        Range:
            0.0 -> 1.0
        """

        archetype_lookup = (
            self._build_archetype_lookup()
        )

        base = (
            users["archetype_id"]
            .map(
                archetype_lookup[
                    "churn_probability"
                ]
            )
            .values
        )

        engagement_penalty = (
            1.0 - engagement_score
        ) * 0.60

        tenure_penalty = (
            1.0 -
            np.clip(
                users["tenure_days"].values
                / 1000,
                0,
                1
            )
        ) * 0.15

        risk = (
            base
            + engagement_penalty
            + tenure_penalty
        )

        return np.clip(
            risk,
            0.01,
            1.0
        )


    def _generate_ltv_score(
        self,
        users: pd.DataFrame,
        engagement_score: np.ndarray
    ) -> np.ndarray:
        """
        Relative lifetime value score.
        """

        tenure_factor = np.clip(
            users["tenure_days"].values
            / 1000,
            0,
            1
        )

        spend_factor = (
            users["monthly_price"].values
            / users["monthly_price"].max()
        )

        ltv = (
            0.40 * engagement_score
            + 0.30 * tenure_factor
            + 0.30 * spend_factor
        )

        return np.clip(
            ltv,
            0,
            1
        )


    # ==================================================
    # SESSION BEHAVIOR
    # ==================================================

    def _generate_avg_session_length(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Average session duration in minutes.
        """

        persona_lookup = (
            self._build_persona_lookup()
        )

        base = (
            users["persona_id"]
            .map(
                persona_lookup[
                    "avg_session_length"
                ]
            )
            .values
        )

        noise = self.rng.normal(
            0,
            10,
            len(users)
        )

        return np.clip(
            base + noise,
            5,
            300
        )


    def _generate_daily_sessions(
        self,
        users: pd.DataFrame,
        engagement_score: np.ndarray
    ) -> np.ndarray:
        """
        Daily session frequency.
        """

        lam = (
            0.5
            + engagement_score * 5
        )

        return np.maximum(
            1,
            self.rng.poisson(
                lam=lam
            )
        )


    def _generate_weekend_ratio(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Weekend activity ratio.
        """

        ratio = self.rng.beta(
            3,
            2,
            len(users)
        )

        return np.clip(
            ratio,
            0.10,
            0.95
        )


    def _generate_night_ratio(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Night viewing ratio.
        """

        age = users["age"].values

        ratio = np.where(
            age < 30,
            self.rng.normal(
                0.60,
                0.15,
                len(users)
            ),
            self.rng.normal(
                0.35,
                0.10,
                len(users)
            )
        )

        return np.clip(
            ratio,
            0.01,
            0.95
        )


    # ==================================================
    # USER STRATEGY METRICS
    # ==================================================

    def _generate_exploration_rate(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        How frequently users consume
        unfamiliar content.
        """

        persona_lookup = (
            self._build_persona_lookup()
        )

        rate = (
            users["persona_id"]
            .map(
                persona_lookup[
                    "exploration_rate"
                ]
            )
            .values
        )

        noise = self.rng.normal(
            0,
            0.05,
            len(users)
        )

        return np.clip(
            rate + noise,
            0.01,
            0.95
        )


    def _generate_loyalty_score(
        self,
        engagement_score: np.ndarray,
        churn_risk: np.ndarray
    ) -> np.ndarray:
        """
        Loyalty toward platform.
        """

        loyalty = (
            engagement_score
            * (1 - churn_risk)
        )

        return np.clip(
            loyalty,
            0,
            1
        )


    # ==================================================
    # SEGMENTATION
    # ==================================================

    def _generate_power_user_flag(
        self,
        engagement_score: np.ndarray,
        avg_daily_sessions: np.ndarray
    ) -> np.ndarray:
        """
        Top platform consumers.
        """

        return (
            (
                engagement_score > 0.80
            )
            &
            (
                avg_daily_sessions >= 4
            )
        )


    def _generate_cold_start_flag(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Newly registered users.
        """

        return (
            users["tenure_days"]
            < 30
        )


    def _generate_marketing_segment(
        self,
        engagement_score: np.ndarray,
        churn_risk: np.ndarray,
        ltv_score: np.ndarray
    ) -> np.ndarray:
        """
        CRM segment labels.
        """

        segments = np.full(
            len(engagement_score),
            "standard",
            dtype=object
        )

        segments[
            ltv_score > 0.80
        ] = "vip"

        segments[
            churn_risk > 0.70
        ] = "retention"

        segments[
            (
                engagement_score > 0.85
            )
            &
            (
                ltv_score > 0.70
            )
        ] = "power"

        return segments


    # ==================================================
    # BEHAVIOR ENRICHMENT
    # ==================================================

    def enrich_behavioral_features(
        self,
        users: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Attach behavioral features.
        """

        engagement_score = (
            self._generate_engagement_score(
                users
            )
        )

        churn_risk = (
            self._generate_churn_risk(
                users,
                engagement_score
            )
        )

        ltv_score = (
            self._generate_ltv_score(
                users,
                engagement_score
            )
        )

        avg_session_length = (
            self._generate_avg_session_length(
                users
            )
        )

        avg_daily_sessions = (
            self._generate_daily_sessions(
                users,
                engagement_score
            )
        )

        weekend_ratio = (
            self._generate_weekend_ratio(
                users
            )
        )

        night_ratio = (
            self._generate_night_ratio(
                users
            )
        )

        exploration_rate = (
            self._generate_exploration_rate(
                users
            )
        )

        loyalty_score = (
            self._generate_loyalty_score(
                engagement_score,
                churn_risk
            )
        )

        users["engagement_score"] = (
            engagement_score
        )

        users["churn_risk"] = (
            churn_risk
        )

        users["lifetime_value_score"] = (
            ltv_score
        )

        users["avg_session_length"] = (
            avg_session_length
        )

        users["avg_daily_sessions"] = (
            avg_daily_sessions
        )

        users["weekend_activity_ratio"] = (
            weekend_ratio
        )

        users["night_activity_ratio"] = (
            night_ratio
        )

        users["exploration_rate"] = (
            exploration_rate
        )

        users["loyalty_score"] = (
            loyalty_score
        )

        users["power_user_flag"] = (
            self._generate_power_user_flag(
                engagement_score,
                avg_daily_sessions
            )
        )

        users["cold_start_flag"] = (
            self._generate_cold_start_flag(
                users
            )
        )

        users["marketing_segment"] = (
            self._generate_marketing_segment(
                engagement_score,
                churn_risk,
                ltv_score
            )
        )

        return users

    # ==================================================
    # PREFERENCE DEFINITIONS
    # ==================================================

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

    DEVICES = [
        "Mobile",
        "Tablet",
        "Desktop",
        "SmartTV",
        "GamingConsole"
    ]

    TASTE_PROFILES = [
        "Mainstream",
        "Blockbuster",
        "Niche",
        "Explorer",
        "Family",
        "BingeWatcher",
        "TrendFollower",
        "PremiumViewer"
    ]


    # ==================================================
    # LANGUAGE PREFERENCES
    # ==================================================

    def _generate_preferred_language(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Preferred content language.
        """

        country_language_map = {
            "USA": "English",
            "UK": "English",
            "Canada": "English",
            "Australia": "English",
            "India": "Hindi",
            "Germany": "German",
            "Brazil": "Portuguese",
            "Japan": "Japanese"
        }

        base_language = (
            users["country"]
            .map(country_language_map)
            .values
        )

        return base_language


    # ==================================================
    # DEVICE OWNERSHIP
    # ==================================================

    def _generate_device_ownership(
        self,
        users: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate realistic device ownership.
        """

        primary_device = []
        secondary_device = []

        for _, row in users.iterrows():

            if row["age"] < 25:

                primary = self.rng.choice(
                    ["Mobile", "Desktop"],
                    p=[0.75, 0.25]
                )

            elif row["age"] < 45:

                primary = self.rng.choice(
                    ["Mobile", "SmartTV", "Desktop"],
                    p=[0.45, 0.35, 0.20]
                )

            else:

                primary = self.rng.choice(
                    ["SmartTV", "Tablet", "Mobile"],
                    p=[0.55, 0.15, 0.30]
                )

            secondary_options = [
                d for d in self.DEVICES
                if d != primary
            ]

            secondary = self.rng.choice(
                secondary_options
            )

            primary_device.append(primary)
            secondary_device.append(secondary)

        return pd.DataFrame(
            {
                "primary_device":
                primary_device,

                "secondary_device":
                secondary_device
            }
        )


    # ==================================================
    # GENRE PREFERENCE SAMPLING
    # ==================================================

    def _sample_genres(
        self,
        n_choices: int
    ) -> List[str]:

        selected = self.rng.choice(
            self.GENRES,
            size=n_choices,
            replace=False
        )

        return list(selected)


    def _generate_preferred_genres(
        self,
        users: pd.DataFrame
    ) -> List[List[str]]:
        """
        Multi-label genre preferences.
        """

        preferences = []

        for _ in range(len(users)):

            n = self.rng.integers(
                2,
                6
            )

            preferences.append(
                self._sample_genres(n)
            )

        return preferences


    # ==================================================
    # GENRE AFFINITY VECTOR
    # ==================================================

    def _generate_genre_affinity_vectors(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Dense preference vector.

        Shape:
            [n_users, n_genres]
        """

        vectors = self.rng.gamma(
            shape=2.0,
            scale=1.0,
            size=(
                len(users),
                len(self.GENRES)
            )
        )

        row_sums = vectors.sum(
            axis=1,
            keepdims=True
        )

        vectors = vectors / row_sums

        return vectors


    # ==================================================
    # CONTENT MATURITY PREFERENCE
    # ==================================================

    def _generate_maturity_preference(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Preferred maturity rating.
        """

        age = users["age"].values

        ratings = []

        for a in age:

            if a < 18:
                ratings.append("Kids")

            elif a < 25:
                ratings.append(
                    self.rng.choice(
                        ["Teen", "Adult"],
                        p=[0.40, 0.60]
                    )
                )

            else:
                ratings.append("Adult")

        return np.array(ratings)


    # ==================================================
    # POPULARITY VS NOVELTY
    # ==================================================

    def _generate_popularity_preference(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Preference for popular content.
        """

        engagement = (
            users["engagement_score"]
            .values
        )

        popularity = (
            0.4
            + engagement * 0.5
            + self.rng.normal(
                0,
                0.1,
                len(users)
            )
        )

        return np.clip(
            popularity,
            0,
            1
        )


    def _generate_novelty_preference(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Preference for discovering
        new content.
        """

        novelty = (
            users["exploration_rate"]
            .values
            + self.rng.normal(
                0,
                0.05,
                len(users)
            )
        )

        return np.clip(
            novelty,
            0,
            1
        )


    # ==================================================
    # EMBEDDING CLUSTERING
    # ==================================================

    def _generate_embedding_cluster(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Lightweight latent cluster assignment.
        """

        n_clusters = 20

        return self.rng.integers(
            0,
            n_clusters,
            len(users)
        )


    # ==================================================
    # TASTE PROFILE
    # ==================================================

    def _generate_taste_profile(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        High-level recommendation persona.
        """

        profile = []

        for _, row in users.iterrows():

            if row["is_premium"]:

                profile.append(
                    self.rng.choice(
                        [
                            "PremiumViewer",
                            "Explorer",
                            "BingeWatcher"
                        ]
                    )
                )

            elif row["age"] < 18:

                profile.append(
                    "Family"
                )

            elif row["exploration_rate"] > 0.50:

                profile.append(
                    "Explorer"
                )

            elif row["engagement_score"] > 0.80:

                profile.append(
                    "BingeWatcher"
                )

            else:

                profile.append(
                    self.rng.choice(
                        [
                            "Mainstream",
                            "TrendFollower",
                            "Blockbuster",
                            "Niche"
                        ]
                    )
                )

        return np.array(profile)


    # ==================================================
    # ENRICH PREFERENCE FEATURES
    # ==================================================

    def enrich_preference_features(
        self,
        users: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Attach preference features.
        """

        users[
            "preferred_language"
        ] = self._generate_preferred_language(
            users
        )

        device_df = (
            self._generate_device_ownership(
                users
            )
        )

        users = pd.concat(
            [
                users,
                device_df
            ],
            axis=1
        )

        users[
            "preferred_genres"
        ] = self._generate_preferred_genres(
            users
        )

        users[
            "content_maturity_preference"
        ] = (
            self._generate_maturity_preference(
                users
            )
        )

        users[
            "popularity_preference"
        ] = (
            self._generate_popularity_preference(
                users
            )
        )

        users[
            "novelty_preference"
        ] = (
            self._generate_novelty_preference(
                users
            )
        )

        users[
            "user_embedding_cluster"
        ] = (
            self._generate_embedding_cluster(
                users
            )
        )

        users[
            "taste_profile"
        ] = (
            self._generate_taste_profile(
                users
            )
        )

        genre_vectors = (
            self._generate_genre_affinity_vectors(
                users
            )
        )

        self.genre_affinity_matrix = (
            genre_vectors
        )

        return users

    # ==================================================
    # HOUSEHOLD PROFILE MODELING
    # ==================================================

    def _generate_household_roles(
        self,
        users: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate profile role inside household.

        Roles:
            Primary
            Adult
            Teen
            Child
        """

        roles = []

        for _, row in users.iterrows():

            age = row["age"]

            if age < 13:
                roles.append("Child")

            elif age < 18:
                roles.append("Teen")

            elif row["household_size"] > 1:

                if self.rng.random() < 0.25:
                    roles.append("Primary")
                else:
                    roles.append("Adult")

            else:
                roles.append("Primary")

        return pd.DataFrame(
            {
                "household_role": roles
            }
        )


    def _generate_family_flags(
        self,
        users: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Family account simulation.
        """

        is_family_account = (
            (
                users["subscription_plan"]
                == "family"
            )
            |
            (
                users["household_size"] >= 3
            )
        )

        is_kids_profile = (
            users["age"] < 13
        )

        return pd.DataFrame(
            {
                "is_family_account":
                is_family_account,

                "is_kids_profile":
                is_kids_profile
            }
        )


    # ==================================================
    # COLD START SIMULATION
    # ==================================================

    def _generate_cold_start_type(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Cold-start category.
        """

        result = np.full(
            len(users),
            "none",
            dtype=object
        )

        recent_mask = (
            users["tenure_days"] <= 30
        )

        result[
            recent_mask
        ] = self.rng.choice(
            [
                "new_user",
                "sparse_user"
            ],
            size=recent_mask.sum(),
            p=[0.7, 0.3]
        )

        return result


    # ==================================================
    # USER QUALITY SCORE
    # ==================================================

    def _generate_user_quality_score(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Internal user quality metric.

        Useful for:
            recommendation simulation
            experimentation
            ranking labels
        """

        quality = (
            users["engagement_score"]
            * 0.35
            +
            users["loyalty_score"]
            * 0.35
            +
            users["lifetime_value_score"]
            * 0.30
        )

        noise = self.rng.normal(
            0,
            0.03,
            len(users)
        )

        quality += noise

        return np.clip(
            quality,
            0,
            1
        )


    # ==================================================
    # USER SEGMENTATION
    # ==================================================

    def _generate_user_segment(
        self,
        users: pd.DataFrame
    ) -> np.ndarray:
        """
        Strategic business segments.
        """

        segments = []

        for _, row in users.iterrows():

            if row["lifetime_value_score"] > 0.80:
                segments.append(
                    "high_value"
                )

            elif row["churn_risk"] > 0.70:
                segments.append(
                    "at_risk"
                )

            elif row["engagement_score"] > 0.80:
                segments.append(
                    "engaged"
                )

            elif row["cold_start_flag"]:
                segments.append(
                    "new_user"
                )

            else:
                segments.append(
                    "standard"
                )

        return np.array(
            segments
        )


    # ==================================================
    # FEATURE STORE EXPORT
    # ==================================================

    def build_user_feature_store(
        self,
        users: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Flattened feature-store table.

        Online/offline feature serving.
        """

        features = users.copy()

        features["event_timestamp"] = (
            pd.Timestamp.utcnow()
        )

        return features


    # ==================================================
    # EMBEDDING JOIN
    # ==================================================

    def attach_user_embeddings(
        self,
        users: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join latent embeddings.

        Produces:
            user_feature_table
        """

        embeddings = (
            self.user_embeddings
            .copy()
        )

        return users.merge(
            embeddings,
            left_on="embedding_id",
            right_on="user_id",
            how="left",
            suffixes=(
                "",
                "_embedding"
            )
        )


    # ==================================================
    # FINAL ENRICHMENT
    # ==================================================

    def enrich_final_features(
        self,
        users: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Final user enrichment.
        """

        roles = (
            self._generate_household_roles(
                users
            )
        )

        family = (
            self._generate_family_flags(
                users
            )
        )

        users = pd.concat(
            [
                users,
                roles,
                family
            ],
            axis=1
        )

        users[
            "cold_start_type"
        ] = (
            self._generate_cold_start_type(
                users
            )
        )

        users[
            "user_quality_score"
        ] = (
            self._generate_user_quality_score(
                users
            )
        )

        users[
            "user_segment"
        ] = (
            self._generate_user_segment(
                users
            )
        )

        return users


    # ==================================================
    # VALIDATION EXTENSIONS
    # ==================================================

    def validate_final(
        self,
        users: pd.DataFrame
    ) -> None:
        """
        Production validation.
        """

        if users.isna().sum().sum() > 0:
            raise ValueError(
                "Missing values detected."
            )

        if (
            users["engagement_score"]
            .min()
            < 0
        ):
            raise ValueError(
                "Invalid engagement score."
            )

        if (
            users["churn_risk"]
            .max()
            > 1
        ):
            raise ValueError(
                "Invalid churn score."
            )

        if (
            users["user_quality_score"]
            .max()
            > 1
        ):
            raise ValueError(
                "Invalid quality score."
            )


    # ==================================================
    # PRODUCTION GENERATE
    # ==================================================

    def generate(
        self
    ) -> pd.DataFrame:
        """
        Production-grade user generation.

        Pipeline
        --------

        Base Users
            ↓

        Behavioral Layer
            ↓

        Preference Layer
            ↓

        Final Layer
            ↓

        Validation
            ↓

        Return
        """

        users = (
            self.generate_base_users()
        )

        users = (
            self.enrich_behavioral_features(
                users
            )
        )

        users = (
            self.enrich_preference_features(
                users
            )
        )

        users = (
            self.enrich_final_features(
                users
            )
        )

        self.validate(
            users
        )

        self.validate_final(
            users
        )

        return users

