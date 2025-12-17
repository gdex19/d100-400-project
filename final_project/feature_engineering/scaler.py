# mypy: disable-error-code=misc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from pandas.api.types import is_numeric_dtype


# Write a BasicScaler, similar to sklearn standard scaler
class BasicScaler(BaseEstimator, TransformerMixin):
    """
    Scaler for sklearn Pipelines. Input must be numeric DataFrames.
    """

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "BasicScaler":
        for dtype in X.dtypes:
            if not is_numeric_dtype(dtype):
                raise TypeError("Passed non-numeric columns")

        self.means_ = X.mean()
        vars = X.var(ddof=0)
        self.variances_ = vars.replace(0.0, 1.0)
        self.columns_ = X.columns

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["means_", "variances_", "columns_"])

        if set(X.columns) != set(self.columns_):
            raise ValueError("Attempted transform with different columns")

        old_columns = X.columns
        X_aligned = X[self.columns_]
        X_new = (X_aligned - self.means_) / (self.variances_**0.5)
        return X_new[old_columns]
