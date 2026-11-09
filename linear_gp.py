"""
Linear GP — Algorithm 1 of Kötzing, Neumann, Spöhel (GECCO 2011)
=================================================================

Algorithm 1 (verbatim from paper):
input: Black-box target function f_OPT;
input: Sample size z;
initialization: Uniformly at random choose an initial function f = Σ wi·xi;
repeat:
    1. Choose i ∈ [n] uniformly at random;
    2. Obtain f' from f by flipping the i-th weight;
    3. Sample a set S ⊆ {0,1}^n of size z;
    4. if eS(f', f_OPT) ≤ eS(f, f_OPT) then f ← f'
until: forever

Error definitions (from paper, Section 2):
    ex(f, f_OPT)  = |f(x) - f_OPT(x)|- error on single point
    eS(f, f_OPT)  = Σ_{x ∈ S} ex(f, f_OPT)- error on a sample set S

Noisy oracle: instead of f_OPT(x), the algorithm observes
    ỹ(x) = f_OPT(x) + ε,   ε ~ N(0, σ²)
"""
import numpy as np
import pandas as pd
from itertools import groupby
from math import comb

def make_target(n: int, rng: np.random.Generator):
    """
    Sample the optimal weight vector w_OPT uniformly from {-1, +1}^n.
    The target function is f_OPT(x) = Σ_{i=1}^{n} w_i · x_i.
    """
    return rng.choice([-1, 1], size=n)

# target linear function f_OPT(x)= Σ_{i=1}^{n} w_i · x_i
def f_opt(x: np.ndarray, w_opt: np.ndarray):
    return float(w_opt @ x)

# generate n bits with random weights belongs to {-1,1}
# Returns array of shape (m, n)
def generate_bitstrings(n: int, m: int, rng: np.random.Generator):
    return rng.integers(0, 2, size=(m, n))

# assumption violation: add gaussian noise to clean targets
def add_gaussian_noise(y_clean:np.ndarray, sigma: float, rng: np.random.Generator):
    if sigma == 0.0:
        return y_clean.copy()
    return y_clean + rng.normal(loc=0.0, scale=sigma, size=y_clean.shape)

# error function: ex(f, f_OPT) = |f(x) - f_OPT(x)| for single point
def ex(x: np.ndarray, w: np.ndarray, y_target: float):
    return abs(float(w @ x) - y_target)

# sample set error function: eS(f, f_OPT)  = Σ_{x ∈ S} ex(f, f_OPT)
def eS(S: np.ndarray, w: np.ndarray, y_S: np.ndarray):
    # shape (z,)
    predictions = S @ w
    return float(np.sum(np.abs(predictions - y_S)))

#evaluation functions: mean squared error
def test_mse(w: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    preds=X_test@w
    return float(np.mean((preds - y_test) ** 2))

# number of weights where w different from w_opt:
def hamming_distance(w: np.ndarray, w_opt: np.ndarray):
    return int(np.sum(w != w_opt))

# pn=0.5 * C(2n, n) * 2^{-2n}
def compute_pn(n: int):
    return 0.5 * comb(2 * n, n) * (2 ** (-2 * n))

# algorithm 1: linear GP for learning functions
def linear_gp(n: int, z: int, w_opt: np.ndarray, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray, max_iter: int, mse_thresh: float,
              rng: np.random.Generator):
    # step 1 initialization: Uniformly at random choose an initial function f = Σ wi·xi
    w = rng.choice([-1, 1], size=n).astype(float)
    recovered = False
    conv_iter = max_iter
    mse_history = []
    weight_diffs = []
    def snapshot(t):
        mse = test_mse(w, X_test, y_test)
        ham = hamming_distance(w, w_opt)
        mse_history.append((t, mse))
        weight_diffs.append((t, ham))
        return mse
    snapshot(0)
    # repeat loop
    for t in range(1, max_iter+1):
        # step 2: Choose i ∈ [n] uniformly at random;
        i = int(rng.integers(0, n))
        # step 3: Obtain f' from f by flipping the i-th weight;
        w_prime = w.copy()
        w_prime[i] *= -1
        # step 4: Sample a set S ⊆ {0,1}^n of size z;
        idx = rng.integers(0, len(X_train), size = z)
        # shape(z,n)
        S = X_train[idx]
        # shape(z,): noisy targets
        y_S = y_train[idx]
        # step 5: if eS(f', f_OPT) ≤ eS(f, f_OPT) then f ← f'
        error_current = eS(S, w, y_S)
        error_prime = eS(S, w_prime, y_S)

        if error_prime <= error_current:
            w=w_prime
        # record snapshot for each 50 iterations
        if t%50==0:
            current_mse=snapshot(t)
            current_mse = snapshot(t)
            if current_mse <= mse_thresh and not recovered:
                recovered = True
                conv_iter = t
    final_mse = test_mse(w, X_test, y_test)
    if final_mse <= mse_thresh and not recovered:
        recovered = True
        conv_iter = max_iter
    return {"w_final":w, "final_mse": final_mse, "recovered": recovered,
            "conv_iter": conv_iter, "mse_history": mse_history, 
            "weight_diffs":weight_diffs}

# run whole experiment
def experiment(n: int=20, c_sample: float=2.0, sigma_list:list = None,
               m_list: list=None, n_seeds: int=20, max_iter: int=None,
               test_size: int=2000, mse_thresh: float=0.1, rng_base: int=42):
    if sigma_list is None:
        sigma_list = [0.0, 0.5, 1.0, 2.0, 4.0]
    if m_list is None:
        m_list = [200, 500, 1000]
    if max_iter is None:
        # 4000 for n=20
        max_iter = 10 * n ** 2
    # sample size: c·n·log(n) ≈ 119
    z = int(c_sample * n * np.log(n))

    print(f"Linear GP — Algorithm 1 (KNS11)")
    print(f"n={n}, z={z}, max_iter={max_iter}, seeds={n_seeds}")
    print(f"σ ∈ {sigma_list}")
    print(f"m ∈ {m_list}\n")

    pn = compute_pn(n)
    print(f"Bias signal: pn = {pn:.6f},  pn/2 = {pn/2:.6f}")
    sigma_critical = (pn / 2) * np.sqrt(z)
    print(f"Theoretical SNR=1 threshold: σ* ≈ {sigma_critical:.4f}\n")

    all_results = []
    total = len(sigma_list) * len(m_list) * n_seeds
    done  = 0
    for sigma in sigma_list:
        for m in m_list:
            for seed in range(n_seeds):
                rng = np.random.default_rng(rng_base+seed*100)
                # true target weights
                w_opt=make_target(n, rng)

                # train data with noisy targets
                X_train = generate_bitstrings(n, m, rng)
                y_clean_tr = X_train @ w_opt
                y_train = add_gaussian_noise(y_clean_tr, sigma, rng)

                # clean test dataset which always noise-free
                X_test = generate_bitstrings(n, test_size, rng)
                y_test = X_test @ w_opt

                # run algorithm 1
                result = linear_gp(n, z, w_opt, X_train, y_train,
                                   X_test, y_test, max_iter, mse_thresh, rng)
                result.update({
                    "sigma": sigma, "m":m, "seed": seed,
                    "n": n, "z": z
                })
                all_results.append(result)

                done+=1
                if done % (n_seeds*len(m_list)) == 0:
                    pct = 100 * done /total
                    print(f"[{pct:5.1f}%] σ={sigma}  "
                          f"({done}/{total} runs complete)")
    print("\n all runs complete.")
    return all_results

# aggregate per seed results into mean ± std per (sigma, m):
def summarise(result: list):
    rows=[]
    key=lambda r:(r["sigma"], r["m"])
    for (sigma, m), grp in groupby(sorted(results, key=key), key=key):
        grp=list(grp)
        mses = [r["final_mse"] for r in grp]
        recovs = [int(r["recovered"]) for r in grp]
        convs = [r["conv_iter"] for r in grp]
        rows.append({
            "sigma": sigma, "m":m, "mean_mse": np.mean(mses),
            "std_mse": np.std(mses), "recovery_pct": np.mean(recovs) * 100,
            "mean_conv": np.mean(convs), "std_conv": np.std(convs)
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":

    results = experiment()
    df = summarise(results)

    print("\nSummary (mean ± std across 20 seeds):")
    print(df.to_string(index=False, float_format="%.3f"))

    df.to_csv("./linear_gp_results.csv", index=False)
    print("\nSaved: linear_gp_results.csv")