
"""
Run paired experiments comparing Linear GP and GP Bagging.

Results are checkpointed after each individual algorithm run.
Completed runs are skipped when the experiment is restarted.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np

from data import make_dataset
from linear_gp import linear_gp, hamming_distance
from gp_bagging import run_gp_bagging


FIELDNAMES = [
    'algorithm',
    'n',
    'm',
    'sigma',
    'seed',
    'z',
    'test_mse',
    'runtime',
    'recovered',
    'conv_iter',
    'weight_hamming',
    'n_nodes',
]

RESULT_KEY_FIELDS = ('algorithm', 'n', 'm', 'sigma', 'seed')


def make_config(
    n: int,
    sigma_list: list[float],
    m_list: list[int],
    n_seeds: int,
    c_sample: float,
    max_iter: int,
    test_size: int,
    mse_thresh: float,
    rng_base: int,
) -> dict:
    """Create metadata used to verify that a checkpoint matches this experiment."""

    return {
        'n': n,
        'sigma_list': sigma_list,
        'm_list': m_list,
        'n_seeds': n_seeds,
        'c_sample': c_sample,
        'max_iter': max_iter,
        'test_size': test_size,
        'mse_thresh': mse_thresh,
        'rng_base': rng_base,
    }


def result_key(row: dict) -> tuple:
    """Return a unique identifier for an algorithm/configuration combination."""

    return (
        row['algorithm'],
        int(row['n']),
        int(row['m']),
        float(row['sigma']),
        int(row['seed']),
    )


def load_checkpoint(filename: str, config: dict) -> tuple[list[dict], set[tuple]]:
    """
    Load existing results and verify their configuration.

    A separate JSON file stores the experiment settings.
    """

    path = Path(filename)
    config_path = Path(f'{filename}.config.json')

    if path.exists() and not config_path.exists():
        raise FileExistsError(
            f'{path} exists, but its configuration file is missing.\n'
            'To avoid mixing incompatible results, rename or move the old CSV '
            'before starting this experiment.'
        )

    if config_path.exists():
        with config_path.open('r') as file:
            saved_config = json.load(file)

        if saved_config != config:
            raise ValueError(
                'The existing checkpoint uses different experiment settings.\n'
                f'Existing configuration: {saved_config}\n'
                f'Current configuration: {config}\n'
                'Use a different results filename or restore the original settings.'
            )

    else:
        path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open('w') as file:
            json.dump(config, file, indent=4)

    if not path.exists():
        return [], set()

    with path.open('r', newline='') as file:
        reader = csv.DictReader(file)
        results = list(reader)

    completed = {result_key(row) for row in results}

    print(f'Loaded {len(results)} existing results from {path}.')

    return results, completed


def save_checkpoint(row: dict, filename: str) -> None:
    """Append one completed algorithm result to the CSV immediately."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with path.open('a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
        file.flush()


def run_linear_experiment(
    dataset,
    n: int,
    m: int,
    sigma: float,
    seed: int,
    z: int,
    max_iter: int,
    mse_thresh: float,
    rng_base: int,
) -> dict:
    """Run Linear GP and return its result row."""

    linear_seed = rng_base + seed * 100 + 1_000_000
    linear_rng = np.random.default_rng(linear_seed)

    start_time = time.perf_counter()

    linear_result = linear_gp(
        n,
        z,
        dataset.w_opt,
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        max_iter,
        mse_thresh,
        linear_rng,
    )

    runtime = time.perf_counter() - start_time

    return {
        'algorithm': 'Linear GP',
        'n': n,
        'm': m,
        'sigma': sigma,
        'seed': seed,
        'z': z,
        'test_mse': linear_result['final_mse'],
        'runtime': runtime,
        'recovered': linear_result['recovered'],
        'conv_iter': linear_result['conv_iter'],
        'weight_hamming': hamming_distance(
            linear_result['w_final'],
            dataset.w_opt,
        ),
        'n_nodes': None,
    }


def run_bagging_experiment(
    dataset,
    n: int,
    m: int,
    sigma: float,
    seed: int,
    rng_base: int,
) -> dict:
    """Run GP Bagging and return its result row."""

    bagging_seed = rng_base + seed * 100 + 2_000_000

    bagging_result = run_gp_bagging(
        dataset=dataset,
        seed=bagging_seed,
    )

    return {
        'algorithm': 'GP Bagging',
        'n': n,
        'm': m,
        'sigma': sigma,
        'seed': seed,
        'z': None,
        'test_mse': bagging_result['final_mse'],
        'runtime': bagging_result['elapsed_time'],
        'recovered': None,
        'conv_iter': None,
        'weight_hamming': None,
        'n_nodes': bagging_result['n_nodes'],
    }


def run_experiments(
    n: int = 20,
    sigma_list: list[float] | None = None,
    m_list: list[int] | None = None,
    n_seeds: int = 20,
    c_sample: float = 2.0,
    max_iter: int | None = None,
    test_size: int = 2000,
    mse_thresh: float = 0.1,
    rng_base: int = 42,
    filename: str = 'results/paired_results.csv',
) -> list[dict]:
    """Run or resume the complete experiment."""

    if sigma_list is None:
        sigma_list = [0.0, 0.5, 1.0, 2.0, 4.0]

    if m_list is None:
        m_list = [200, 500, 1000]

    if max_iter is None:
        max_iter = 10 * n ** 2

    config = make_config(
        n=n,
        sigma_list=sigma_list,
        m_list=m_list,
        n_seeds=n_seeds,
        c_sample=c_sample,
        max_iter=max_iter,
        test_size=test_size,
        mse_thresh=mse_thresh,
        rng_base=rng_base,
    )

    all_results, completed = load_checkpoint(filename, config)

    z = int(c_sample * n * np.log(n))

    total = len(sigma_list) * len(m_list) * n_seeds * 2
    skipped = len(completed)
    run_count = 0

    print('\nLinear GP vs. GP Bagging')
    print(f'n={n}, max_iter={max_iter}, seeds={n_seeds}')
    print(f'Noise levels: {sigma_list}')
    print(f'Training sizes: {m_list}')
    print(f'Total algorithm runs: {total}')
    print(f'Already completed: {skipped}')
    print(f'Remaining: {total - skipped}\n')

    for sigma in sigma_list:
        for m in m_list:
            for seed in range(n_seeds):

                # Generate the shared dataset deterministically.
                dataset_seed = rng_base + seed * 100

                dataset = make_dataset(
                    n=n,
                    m=m,
                    test_size=test_size,
                    sigma=sigma,
                    seed=dataset_seed,
                )

                # --------------------------------------------------
                # 1. Linear GP
                # --------------------------------------------------

                key = ('Linear GP', n, m, float(sigma), seed)

                if key not in completed:
                    print(
                        f'Running Linear GP: sigma={sigma}, '
                        f'm={m}, seed={seed}'
                    )

                    row = run_linear_experiment(
                        dataset=dataset,
                        n=n,
                        m=m,
                        sigma=sigma,
                        seed=seed,
                        z=z,
                        max_iter=max_iter,
                        mse_thresh=mse_thresh,
                        rng_base=rng_base,
                    )

                    save_checkpoint(row, filename)
                    all_results.append(row)
                    completed.add(key)
                    run_count += 1

                    print(
                        f"  Test MSE: {row['test_mse']:.6f} "
                        f"| Runtime: {row['runtime']:.2f}s"
                    )
                else:
                    print(
                        f'Skipping completed Linear GP: '
                        f'sigma={sigma}, m={m}, seed={seed}'
                    )

                # --------------------------------------------------
                # 2. GP Bagging
                # --------------------------------------------------

                key = ('GP Bagging', n, m, float(sigma), seed)

                if key not in completed:
                    print(
                        f'Running GP Bagging: sigma={sigma}, '
                        f'm={m}, seed={seed}'
                    )

                    row = run_bagging_experiment(
                        dataset=dataset,
                        n=n,
                        m=m,
                        sigma=sigma,
                        seed=seed,
                        rng_base=rng_base,
                    )

                    save_checkpoint(row, filename)
                    all_results.append(row)
                    completed.add(key)
                    run_count += 1

                    print(
                        f"  Test MSE: {row['test_mse']:.6f} "
                        f"| Runtime: {row['runtime']:.2f}s"
                    )
                else:
                    print(
                        f'Skipping completed GP Bagging: '
                        f'sigma={sigma}, m={m}, seed={seed}'
                    )

                print(
                    f'Progress: {len(completed)}/{total} algorithm runs '
                    f'({100 * len(completed) / total:.1f}%)\n'
                )

    print('All experiments complete!')
    print(f'New algorithm runs this session: {run_count}')
    print(f'Total saved algorithm runs: {len(completed)}')
    print(f'Results: {filename}')

    return all_results


if __name__ == '__main__':

    # Full experiment configuration.
    results = run_experiments(
        n=20,
        sigma_list=[0.0], #, 0.5, 1.0, 2.0, 4.0],
        m_list=[200], #, 500, 1000],
        n_seeds=1, #20,
    )
