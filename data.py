from dataclasses import dataclass

import numpy as np

@dataclass
class SyntheticDataset:
    """A generated training and test dataset for the linear target function."""

    w_opt: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def make_target(n: int, rng: np.random.Generator):
    """
    Sample the optimal weight vector w_OPT uniformly from {-1, +1}^n.
    The target function is f_OPT(x) = Σ_{i=1}^{n} w_i · x_i.
    """
    return rng.choice([-1, 1], size=n)


# generate n bits with random weights belongs to {-1,1}
# Returns array of shape (m, n)
def generate_bitstrings(n: int, m: int, rng: np.random.Generator):
    return rng.integers(0, 2, size=(m, n)).astype(float)


# target linear function f_OPT(x)= Σ_{i=1}^{n} w_i · x_i
def f_opt(x: np.ndarray, w_opt: np.ndarray):
    return x @ w_opt


# assumption violation: add gaussian noise to clean targets
def add_gaussian_noise(y_clean:np.ndarray, sigma: float, rng: np.random.Generator):
    if sigma == 0.0:
        return y_clean.copy()
    return y_clean + rng.normal(loc=0.0, scale=sigma, size=y_clean.shape)


def make_dataset(n: int, m: int, test_size: int, sigma: float, seed: int):
    """Generate a dataset with noisy training targets and clean test targets."""
    rng = np.random.default_rng(seed)

    # Generate the true target function
    w_opt = make_target(n, rng)

    # Generate training inputs and their clean target values
    X_train = generate_bitstrings(n, m, rng)
    y_clean_train = f_opt(X_train, w_opt)

    # Generate the clean test set before generating noise
    X_test = generate_bitstrings(n, test_size, rng)
    y_test = f_opt(X_test, w_opt)

    # Generate the noise even when sigma is zero
    # Keeps random number sequence consistent across noise levels
    noise = rng.normal(loc=0.0, scale=1.0, size=m)
    y_train = y_clean_train + sigma * noise


    return SyntheticDataset(
        w_opt = w_opt,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
    )