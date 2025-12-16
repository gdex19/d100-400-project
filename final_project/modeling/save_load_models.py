import json
from pathlib import Path
from typing import Any, Dict, Tuple
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from final_project.modeling.pipelines import (
    get_glm_pipeline,
    get_lgbm_pipeline,
)


def save_models(
    glm_params: Dict[str, Any], lgbm_params: Dict[str, Any]
) -> None:
    """
    Save best hyperparams for GLM and LGBM.

    Parameters
    ----------
    glm_params : Dict[str, Any]
        Best hyperparams for glm.

    lgbm_params : Dict[str, Any]
        Best hyperparams for lgbm.

    Returns
    -------
    None
    """
    model_dir = Path(__file__).parent.parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "glm": glm_params,
        "lgbm": lgbm_params,
    }

    with open(model_dir / "best_params.json", "w") as f:
        json.dump(params, f, indent=2, default=str)


def load_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[Pipeline, Pipeline]:
    """
    Load best hyperparameters from disk, rebuild pipelines, fit on
    training data, and return fitted (glm, lgbm).

    Parameters
    ----------
    X_train : pd.DataFrame
        Features from training script.

    y_train : pd.DataFrame
        Responders from training script.

    Returns
    -------
    Tuple[Pipeline, Pipeline]
        Fitted glm and lgbm.
    """

    model_dir = Path(__file__).parent.parent.parent / "models"

    with open(model_dir / "best_params.json", "r") as f:
        params = json.load(f)

    glm_params = params["glm"]
    lgbm_params = params["lgbm"]

    # Fix passthrough bug
    key = "preprocess__log_num__log"
    val = glm_params.get(key)

    if val in (None, "passthrough"):
        glm_params[key] = "passthrough"
    else:
        glm_params[key] = FunctionTransformer(
            np.log1p, feature_names_out="one-to-one"
        )

    glm = get_glm_pipeline().set_params(**glm_params)
    lgbm = get_lgbm_pipeline().set_params(**lgbm_params)

    glm.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    return glm, lgbm
