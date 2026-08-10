import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s"
)

project_name = "recommendation_system"

list_of_files = [

    # --------------------------------------------------
    # GitHub
    # --------------------------------------------------
    ".github/workflows/.gitkeep",

    # --------------------------------------------------
    # Configs
    # --------------------------------------------------
    "configs/generator.yaml",
    "configs/retrieval.yaml",
    "configs/ranking.yaml",
    "configs/reranking.yaml",
    "configs/sequential.yaml",
    "configs/graph.yaml",
    "configs/bandit.yaml",
    "configs/rl.yaml",
    "configs/mlflow.yaml",
    "configs/monitoring.yaml",
    "configs/deployment.yaml",

    # --------------------------------------------------
    # Root Files
    # --------------------------------------------------
    "README.md",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    ".gitignore",
    ".env",
    "Makefile",
    "dvc.yaml",
    "params.yaml",

    # --------------------------------------------------
    # Research
    # --------------------------------------------------
    "research/01_end_to_end_recommendation_system.ipynb",

    # --------------------------------------------------
    # Templates
    # --------------------------------------------------
    "templates/index.html",

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
    "data/training/.gitkeep",
    "data/feature_store/.gitkeep",
    "data/monitoring/.gitkeep",

    # --------------------------------------------------
    # Artifacts
    # --------------------------------------------------
    "artifacts/models/.gitkeep",
    "artifacts/embeddings/.gitkeep",
    "artifacts/metrics/.gitkeep",
    "artifacts/reports/.gitkeep",

    # --------------------------------------------------
    # Main Package
    # --------------------------------------------------
    f"src/{project_name}/__init__.py",

    # ==================================================
    # COMMON
    # ==================================================
    f"src/{project_name}/common/__init__.py",
    f"src/{project_name}/common/constants.py",
    f"src/{project_name}/common/enums.py",
    f"src/{project_name}/common/exceptions.py",
    f"src/{project_name}/common/schemas.py",
    f"src/{project_name}/common/utilities.py",

    # ==================================================
    # CONFIG
    # ==================================================
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/config/entities.py",

    # ==================================================
    # FOUNDATION
    # ==================================================
    f"src/{project_name}/foundation/__init__.py",
    f"src/{project_name}/foundation/persona_engine.py",
    f"src/{project_name}/foundation/archetype_engine.py",
    f"src/{project_name}/foundation/latent_factor_engine.py",
    f"src/{project_name}/foundation/foundation_builder.py",

    # ==================================================
    # GENERATORS
    # ==================================================
    f"src/{project_name}/generators/__init__.py",

    f"src/{project_name}/generators/users/__init__.py",
    f"src/{project_name}/generators/users/user_generator.py",

    f"src/{project_name}/generators/items/__init__.py",
    f"src/{project_name}/generators/items/item_generator.py",

    f"src/{project_name}/generators/contexts/__init__.py",
    f"src/{project_name}/generators/contexts/context_generator.py",

    f"src/{project_name}/generators/sessions/__init__.py",
    f"src/{project_name}/generators/sessions/session_generator.py",

    f"src/{project_name}/generators/interactions/__init__.py",
    f"src/{project_name}/generators/interactions/interaction_generator.py",

    f"src/{project_name}/generators/feedback/__init__.py",
    f"src/{project_name}/generators/feedback/review_generator.py",

    f"src/{project_name}/generators/streams/__init__.py",
    f"src/{project_name}/generators/streams/event_stream_generator.py",

    # ==================================================
    # AFFINITY
    # ==================================================
    f"src/{project_name}/affinity/__init__.py",
    f"src/{project_name}/affinity/affinity_engine.py",

    # ==================================================
    # BIAS
    # ==================================================
    f"src/{project_name}/bias/__init__.py",
    f"src/{project_name}/bias/position_bias_engine.py",
    f"src/{project_name}/bias/context_bias_engine.py",

    # ==================================================
    # SEARCH
    # ==================================================
    f"src/{project_name}/search/__init__.py",
    f"src/{project_name}/search/query_generator.py",
    f"src/{project_name}/search/search_dataset_builder.py",

    # ==================================================
    # RETRIEVAL
    # ==================================================
    f"src/{project_name}/retrieval/__init__.py",
    f"src/{project_name}/retrieval/two_tower_dataset.py",
    f"src/{project_name}/retrieval/retrieval_builder.py",

    # ==================================================
    # RANKING
    # ==================================================
    f"src/{project_name}/ranking/__init__.py",
    f"src/{project_name}/ranking/ranking_dataset.py",

    # ==================================================
    # RERANKING
    # ==================================================
    f"src/{project_name}/reranking/__init__.py",
    f"src/{project_name}/reranking/reranking_dataset.py",

    # ==================================================
    # BANDITS
    # ==================================================
    f"src/{project_name}/bandits/__init__.py",
    f"src/{project_name}/bandits/contextual_bandit_dataset.py",

    # ==================================================
    # RL
    # ==================================================
    f"src/{project_name}/reinforcement_learning/__init__.py",
    f"src/{project_name}/reinforcement_learning/rl_dataset_builder.py",

    # ==================================================
    # SEQUENTIAL
    # ==================================================
    f"src/{project_name}/sequential/__init__.py",
    f"src/{project_name}/sequential/sasrec_dataset.py",
    f"src/{project_name}/sequential/bert4rec_dataset.py",

    # ==================================================
    # GRAPH
    # ==================================================
    f"src/{project_name}/graph/__init__.py",
    f"src/{project_name}/graph/graph_builder.py",
    f"src/{project_name}/graph/lightgcn_dataset.py",

    # ==================================================
    # FEATURES
    # ==================================================
    f"src/{project_name}/features/__init__.py",
    f"src/{project_name}/features/user_features.py",
    f"src/{project_name}/features/item_features.py",
    f"src/{project_name}/features/context_features.py",

    # ==================================================
    # FEATURE STORE
    # ==================================================
    f"src/{project_name}/feature_store/__init__.py",
    f"src/{project_name}/feature_store/entities.py",
    f"src/{project_name}/feature_store/feature_views.py",

    # ==================================================
    # DATASETS
    # ==================================================
    f"src/{project_name}/datasets/__init__.py",
    f"src/{project_name}/datasets/training_dataset_builder.py",

    # ==================================================
    # VALIDATION
    # ==================================================
    f"src/{project_name}/validation/__init__.py",
    f"src/{project_name}/validation/data_validation.py",
    f"src/{project_name}/validation/quality_checks.py",

    # ==================================================
    # ANALYTICS
    # ==================================================
    f"src/{project_name}/analytics/__init__.py",
    f"src/{project_name}/analytics/recommendation_analysis.py",

    # ==================================================
    # MODELS
    # ==================================================
    f"src/{project_name}/models/__init__.py",

    # ==================================================
    # EVALUATION
    # ==================================================
    f"src/{project_name}/evaluation/__init__.py",
    f"src/{project_name}/evaluation/retrieval_metrics.py",
    f"src/{project_name}/evaluation/ranking_metrics.py",
    f"src/{project_name}/evaluation/business_metrics.py",

    # ==================================================
    # MONITORING
    # ==================================================
    f"src/{project_name}/monitoring/__init__.py",
    f"src/{project_name}/monitoring/drift_detection.py",
    f"src/{project_name}/monitoring/dashboard.py",

    # ==================================================
    # MLOPS
    # ==================================================
    f"src/{project_name}/mlops/__init__.py",
    f"src/{project_name}/mlops/mlflow_manager.py",
    f"src/{project_name}/mlops/model_registry.py",

    # ==================================================
    # SERVING
    # ==================================================
    f"src/{project_name}/serving/__init__.py",
    f"src/{project_name}/serving/recommendation_service.py",

    # ==================================================
    # API
    # ==================================================
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/app.py",

    # ==================================================
    # ORCHESTRATOR
    # ==================================================
    f"src/{project_name}/orchestrator/__init__.py",
    f"src/{project_name}/orchestrator/recommendation_data_generator.py",
    f"src/{project_name}/orchestrator/master_orchestrator.py",

    # ==================================================
    # TESTS
    # ==================================================
    "tests/unit/.gitkeep",
    "tests/integration/.gitkeep",

    # ==================================================
    # DOCKER
    # ==================================================
    "docker/Dockerfile",
    "docker/docker-compose.yml",

    # ==================================================
    # KUBERNETES
    # ==================================================
    "kubernetes/deployment.yaml",
    "kubernetes/service.yaml",

    # ==================================================
    # SCRIPTS
    # ==================================================
    "scripts/generate_data.py",
    "scripts/train_models.py",
    "scripts/evaluate_models.py",
    "scripts/deploy_models.py",
]

for filepath in list_of_files:

    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

        logging.info(
            f"Creating directory: {filedir}"
        )

    if (not os.path.exists(filepath) or os.path.getsize(filepath) == 0):
        with open(filepath, "w"):
            pass

        logging.info(
            f"Creating empty file: {filepath}"
        )

    else:
        logging.info(
            f"{filename} already exists"
        )