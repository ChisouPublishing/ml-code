import json
import platform
import sys
import warnings

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info


def main():
    print("Python:", sys.version)
    print("Executable:", sys.executable)
    print("OS:", platform.platform())
    print("NumPy:", np.__version__)
    print("BLAS:", json.dumps(threadpool_info(), ensure_ascii=False))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(divide="raise", over="raise", invalid="raise"):
            for size in [15, 784]:
                identity = np.eye(size)
                np.testing.assert_array_equal(identity @ identity, identity)
            generator = np.random.default_rng(42)
            features = generator.normal(size=(4128, 8))
            weights = generator.normal(size=8)
            np.testing.assert_allclose(
                features @ weights,
                np.einsum("ij,j->i", features, weights, optimize=False),
                rtol=1e-12, atol=1e-12)
            dataset, targets = load_diabetes(return_X_y=True)
            train, test, train_targets, _ = train_test_split(
                dataset, targets, random_state=42)
            scaler = StandardScaler().fit(train)
            model = LinearRegression().fit(scaler.transform(train), train_targets)
            test_scaled = scaler.transform(test)
            prediction = model.predict(test_scaled)
            reference = np.einsum("ij,j->i", test_scaled, model.coef_, optimize=False) + model.intercept_
            assert np.isfinite(prediction).all()
            np.testing.assert_allclose(prediction, reference, rtol=1e-12, atol=1e-12)
            print("LinearRegression maximum difference:", np.max(np.abs(prediction - reference)))
            try:
                np.divide(1.0, 0.0)
            except FloatingPointError:
                print("Real divide-by-zero: correctly detected")
            else:
                raise AssertionError("Floating-point error detection is disabled")
    print("PASS: matrix multiplication, finite predictions, reference comparison, error detection")


if __name__ == "__main__":
    main()
