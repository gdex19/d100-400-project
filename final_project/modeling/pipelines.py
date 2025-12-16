from glum import GeneralizedLinearRegressor, GammaDistribution
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from final_project.preprocessing import NUM_FEATURES, CAT_FEATURES


def get_glm_pipeline() -> Pipeline:
    """
    Create glm pipeline.

    Returns
    -------
    Pipeline
        glm pipeline.
    """
    LINEAR_NUM_FEATURES = [
        f
        for f in NUM_FEATURES
        if any(k in f.lower() for k in ["m_ret", "pct_change"])
    ]
    LOG_NUM_FEATURES = [
        f for f in NUM_FEATURES if f not in LINEAR_NUM_FEATURES
    ]

    num_log_pipeline = Pipeline(
        steps=[
            ("log", "passthrough"),
            ("scale", StandardScaler()),
        ]
    )

    preprocessor_glm = ColumnTransformer(
        transformers=[
            ("log_num", num_log_pipeline, LOG_NUM_FEATURES),
            (
                "lin_num",
                Pipeline([("scale", StandardScaler())]),
                LINEAR_NUM_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                CAT_FEATURES,
            ),
        ],
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor_glm),
            (
                "glm",
                GeneralizedLinearRegressor(
                    family=GammaDistribution(),
                    link="log",
                    max_iter=10_000,
                ),
            ),
        ]
    )


def get_lgbm_pipeline() -> Pipeline:
    """
    Create LightGBM pipeline.

    Returns
    -------
    Pipeline
        lgbm pipeline.
    """
    preprocessor_lgbm = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                CAT_FEATURES,
            ),
        ],
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor_lgbm),
            (
                "lgbm",
                LGBMRegressor(objective="gamma", random_state=42, n_jobs=1),
            ),
        ]
    )
