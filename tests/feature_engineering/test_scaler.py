# mypy: disable-error-code=misc

import numpy as np
import pytest
import pandas as pd
from sklearn.exceptions import NotFittedError
from final_project.feature_engineering import BasicScaler


@pytest.mark.parametrize("m, n", [(100, 1), (1000, 3), (500, 5)])
def test_standardization(m: int, n: int) -> None:
    """
    Test that standardization works as expected.
    """
    np.random.seed(42)
    X = pd.DataFrame(np.random.normal(size=(m, n)))

    scaler = BasicScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # Make sure mean close to 0
    assert np.allclose(X_scaled.mean(), 0.0, atol=1e-10)

    # Make sure standard close to 1
    assert np.allclose(X_scaled.std(ddof=0), 1.0, atol=1e-10)


def test_constant_column() -> None:
    """
    Test that constant columns do not cause errors.
    """
    X = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0],
            "b": [0.0, 2.0, 4.0],
        }
    )

    scaler = BasicScaler().fit(X)
    X_scaled = scaler.transform(X)

    # constant becomes 0
    assert np.allclose(X_scaled["a"], 0.0)

    # Other column is mean 0 std 1
    assert np.isclose(X_scaled["b"].std(ddof=0), 1.0)
    assert np.isclose(X_scaled["b"].mean(), 0.0)


def test_unfitted() -> None:
    """
    Test handling of calling transform before calling fit.
    """
    X = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0],
            "b": [0.0, 2.0, 4.0],
        }
    )

    scaler = BasicScaler()
    with pytest.raises(NotFittedError):
        scaler.transform(X)


def test_non_numeric() -> None:
    """
    Test the handling of non-numeric data.
    """
    X = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})

    scaler = BasicScaler()
    with pytest.raises(TypeError):
        scaler.fit(X)


@pytest.mark.parametrize("c1, c2", [("b", "a"), ("b", "c")])
def test_column_names(c1: str, c2: str) -> None:
    """
    Test handling of different columns names or ordering
    in fit and transform.
    """
    X1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    # Change order
    X2 = pd.DataFrame({c1: [3.0, 4.0], c2: [1.0, 2.0]})

    scaler = BasicScaler().fit(X1)

    if c2 == "c":
        with pytest.raises(ValueError):
            scaler.transform(X2)
    else:
        X_scaled = scaler.transform(X2)
        assert np.allclose(X_scaled.mean(), 0.0, atol=1e-10)
        assert np.allclose(X_scaled.std(ddof=0), 1.0, atol=1e-10)
