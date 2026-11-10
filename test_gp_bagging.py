import numpy as np

from data import make_dataset
from gp_bagging import run_gp_bagging


dataset = make_dataset(
    n=20,
    m=100,
    test_size=200,
    sigma=1.0,
    seed=42,
)

result = run_gp_bagging(
    dataset=dataset,
    seed=123,
)

print(f"Test MSE: {result['final_mse']:.4f}")
print(f"Runtime: {result['elapsed_time']:.2f} seconds")
print(f"Ensemble nodes: {result['n_nodes']}")
print(f"Predictions shape: {result['predictions'].shape}")

assert result['predictions'].shape == (200,)
assert np.isfinite(result['final_mse'])