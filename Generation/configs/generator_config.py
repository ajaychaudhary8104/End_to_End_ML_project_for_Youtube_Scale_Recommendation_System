from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """
    Global configuration for synthetic recommendation
    ecosystem generation.
    """

    n_users: int = 100_000
    n_items: int = 10_000

    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"

    latent_dim: int = 64
    embedding_dim: int = 64

    random_state: int = 42

    output_dir: str = "artifacts"

    avg_sessions_per_user: int = 20
    avg_interactions_per_user: int = 100

    @classmethod
    def small(cls):
        return cls(
            n_users=10_000,
            n_items=2_000
        )

    @classmethod
    def medium(cls):
        return cls(
            n_users=100_000,
            n_items=10_000
        )

    @classmethod
    def large(cls):
        return cls(
            n_users=1_000_000,
            n_items=100_000
        )