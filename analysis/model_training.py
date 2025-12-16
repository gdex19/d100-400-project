# %%
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    RandomizedSearchCV,
)
from final_project.data import read_data, split_data
from final_project.preprocessing import NUM_FEATURES, CAT_FEATURES, RESPONDER
from final_project.plotting import plot_lasso_paths_numeric
from final_project.modeling import (
    fit_glm_lasso_path,
    save_models,
    get_glm_pipeline,
    get_lgbm_pipeline,
)

# %%
df = read_data("clean_data")

df.head()
# %%
# Perform train/test/val split: 0.6, 0, 0.4
split = [0.6, 0, 0.4]
df_train, df_test, df_val = split_data(df, split)
print(f"Lengths {len(df_train)}, {len(df_test)}, {len(df_val)}")
# %%
# First step is feature selection. I'm going to use
# LASSO to build some intuition here.

lasso_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False),
            CAT_FEATURES,
        ),
    ],
)
lasso_preprocessor.set_output(transform="pandas")


# %%
y = df_train[RESPONDER]
X = df_train[NUM_FEATURES + CAT_FEATURES]
alphas = np.logspace(-4, 1, 40)

coef_path = fit_glm_lasso_path(X, y, lasso_preprocessor, alphas)
# %%
fig = plot_lasso_paths_numeric(coef_path)

# %%
# Now we make our glm pipeline. We will log-transform the positive,
# right-skewed variables, and scale the rest. The pipe is defined
# in the modeling module.

# TODO: add clipped features?
glm_pipe = get_glm_pipeline()

log_step = FunctionTransformer(np.log1p, feature_names_out="one-to-one")

glm_param_grid = {
    # log-transform on/off
    "preprocess__log_num__log": ["passthrough", log_step],
    # regularization from project description
    "glm__alpha": np.logspace(-6, -1, 10),
    "glm__l1_ratio": [
        0.01,
        0.05,
        0.1,
        0.15,
        0.3,
    ],  # 0.1 was best, remove high L1's
}

# Forward-looking CV
tscv = TimeSeriesSplit(n_splits=5)

glm_cv = GridSearchCV(
    glm_pipe,
    param_grid=glm_param_grid,
    cv=tscv,
    scoring="neg_mean_gamma_deviance",
    error_score=np.nan,
    verbose=2,
)

# %%
# Make lgbm pipeline
lgbm_pipe = get_lgbm_pipeline()

lgbm_param_dist = {
    "lgbm__learning_rate": [0.005, 0.01, 0.05, 0.1],
    "lgbm__num_leaves": [15, 31, 63, 127],
    "lgbm__min_child_weight": [0.01, 0.02, 0.05, 0.1],
    "lgbm__n_estimators": [50, 100, 200, 400],
    "lgbm__max_depth": [7],
}

lgbm_cv = RandomizedSearchCV(
    lgbm_pipe,
    param_distributions=lgbm_param_dist,
    n_iter=200,
    cv=tscv,
    scoring="neg_mean_gamma_deviance",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

# %%

X_train = df_train[NUM_FEATURES + CAT_FEATURES]
y_train = df_train[RESPONDER]

glm_cv.fit(X_train, y_train)

print("GLM best:", glm_cv.best_params_, glm_cv.best_score_)
# %%
lgbm_cv.fit(X_train, y_train)
print("LGBM best:", lgbm_cv.best_params_, lgbm_cv.best_score_)

# %%
# Save models to disk
# directory to save models
save_models(glm_cv.best_params_, lgbm_cv.best_params_)
# %%
