from glum import GeneralizedLinearRegressor, GammaDistribution
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from typing import cast
from numpy.typing import NDArray
from final_project.preprocessing import NUM_FEATURES


def fit_glm_lasso_path(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    alphas: NDArray[np.floating],
) -> pd.DataFrame:
    """
    Fit L1 regularized gamma GLMs to perform feature selection.
    Only penalize numerical predictors.

    Parameters
    ----------
    X : pd.DataFrame
        All features.

    y : pd.Series
        Response variable.

    preprocessor : ColumnTransformer
        Preprocessing transformer, standardizes numeric features
        and one-hot encodes categorical features.

    alphas : NDArray[np.floating]
        Sequence of L1 regularizations.

    Returns
    -------
    pd.DataFrame
        DataFrame of coefficient paths indexed by alpha, with one
        column per transformed feature.
    """
    coef_path = []

    X_tr: pd.DataFrame = cast(pd.DataFrame, preprocessor.fit_transform(X))
    feature_names = X_tr.columns

    BOOL_FEATURES = [c for c in NUM_FEATURES if c.startswith("is_")]
    CONT_NUM_FEATURES = [c for c in NUM_FEATURES if c not in BOOL_FEATURES]

    p1 = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        if name.startswith("num__"):
            raw = name.replace("num__", "")
            if raw in CONT_NUM_FEATURES:
                p1[i] = 1.0

    for alpha in alphas:
        glm = GeneralizedLinearRegressor(
            family=GammaDistribution(),
            link="log",
            alpha=alpha,
            l1_ratio=1,
            fit_intercept=True,
            P1=p1,
        )

        glm.fit(X_tr, y)

        coefs = pd.Series(glm.coef_, index=feature_names, name=alpha)
        coef_path.append(coefs)

    coef_df: pd.DataFrame = pd.concat(coef_path, axis=1).T
    coef_df.index.name = "alpha"

    # Drop boolean features
    bool_cols = [c for c in coef_df.columns if c.startswith("num__is_")]

    coef_df = coef_df.drop(columns=bool_cols)

    return coef_df
