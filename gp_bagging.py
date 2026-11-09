"""GP Bagging wrapper using the paper's 2SEGP implementation"""

import sys
import time
import numpy as np

from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import (
    AddNode,
    SubNode,
    MulNode,
    DivNode,
    LogNode,
)
from ssegp.SKLearnInterface import SSEGPRegressionEstimator
from data import SyntheticDataset


# Allow deep expression trees to be copied.
sys.setrecursionlimit(100000)


class SqrtNode(Node):
    """Square root node, protected against negative inputs."""

    def __init__(self):
        super().__init__()
        self.arity = 1

    def __repr__(self):
        return 'sqrt'

    def GetOutput(self, X):
        X0 = self._children[0].GetOutput(X)
        return np.sqrt(np.abs(X0))


def make_gp_bagging_model():
    """Create a GP Bagging regression model using the paper's settings."""

    return SSEGPRegressionEstimator(
        pop_size=500,
        ensemble_size=50,
        max_generations=100,
        verbose=False,
        max_tree_size=500,
        crossover_rate=0.5,
        mutation_rate=0.5,
        initialization_max_tree_height=6,
        use_linear_scaling=True,
        use_erc=True,
        error_metric='mse',
        functions=[
            AddNode(),
            SubNode(),
            MulNode(),
            DivNode(),
            LogNode(),
            SqrtNode(),
        ],
    )


def test_mse(
    predictions: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Calculate mean squared error against the clean test targets."""

    return float(np.mean((predictions - y_test) ** 2))


def run_gp_bagging(
    dataset: SyntheticDataset,
    seed: int,
) -> dict:
    """Train GP Bagging and evaluate it on a shared synthetic dataset."""

    # 2SEGP uses NumPy's global random-number generator internally.
    np.random.seed(seed)

    model = make_gp_bagging_model()

    # Train using the noisy training labels.
    start_time = time.perf_counter()
    model.fit(dataset.X_train, dataset.y_train)
    elapsed_time = time.perf_counter() - start_time

    # Evaluate against the clean test targets.
    predictions = model.predict(dataset.X_test)
    final_mse = test_mse(predictions, dataset.y_test)

    # Count the nodes in the final ensemble.
    ensemble = model.get_ensemble_elitists()
    n_nodes = sum(
        len(individual.GetSubtree())
        for individual in ensemble
    )

    return {
        'final_mse': final_mse,
        'elapsed_time': elapsed_time,
        'n_nodes': n_nodes,
        'predictions': predictions,
        'model': model,
    }