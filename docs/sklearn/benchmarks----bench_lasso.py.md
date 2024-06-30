# `D:\src\scipysrc\scikit-learn\benchmarks\bench_lasso.py`

```
"""
Benchmarks of Lasso vs LassoLars

First, we fix a training set and increase the number of
samples. Then we plot the computation time as function of
the number of samples.

In the second benchmark, we increase the number of dimensions of the
training set. Then we plot the computation time as function of
the number of dimensions.

In both cases, only 10% of the features are informative.
"""

import gc
from time import time

import numpy as np

from sklearn.datasets import make_regression


def compute_bench(alpha, n_samples, n_features, precompute):
    lasso_results = []
    lars_lasso_results = []

    it = 0

    # Iterate over different numbers of samples and features
    for ns in n_samples:
        for nf in n_features:
            it += 1
            print("==================")
            print("Iteration %s of %s" % (it, max(len(n_samples), len(n_features))))
            print("==================")

            # Determine number of informative features
            n_informative = nf // 10
            
            # Generate synthetic regression dataset
            X, Y, coef_ = make_regression(
                n_samples=ns,
                n_features=nf,
                n_informative=n_informative,
                noise=0.1,
                coef=True,
            )

            # Normalize data (each feature column)
            X /= np.sqrt(np.sum(X**2, axis=0))

            # Perform garbage collection
            gc.collect()

            # Benchmarking Lasso
            print("- benchmarking Lasso")
            clf = Lasso(alpha=alpha, fit_intercept=False, precompute=precompute)
            tstart = time()
            clf.fit(X, Y)
            lasso_results.append(time() - tstart)

            # Perform garbage collection
            gc.collect()

            # Benchmarking LassoLars
            print("- benchmarking LassoLars")
            clf = LassoLars(alpha=alpha, fit_intercept=False, precompute=precompute)
            tstart = time()
            clf.fit(X, Y)
            lars_lasso_results.append(time() - tstart)

    return lasso_results, lars_lasso_results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from sklearn.linear_model import Lasso, LassoLars

    alpha = 0.01  # regularization parameter

    n_features = 10
    list_n_samples = np.linspace(100, 1000000, 5).astype(int)

    # Benchmarking with varying number of samples and fixed number of features
    lasso_results, lars_lasso_results = compute_bench(
        alpha, list_n_samples, [n_features], precompute=True
    )

    # Plotting results for the first benchmark
    plt.figure("scikit-learn LASSO benchmark results")
    plt.subplot(211)
    plt.plot(list_n_samples, lasso_results, "b-", label="Lasso")
    plt.plot(list_n_samples, lars_lasso_results, "r-", label="LassoLars")
    plt.title("precomputed Gram matrix, %d features, alpha=%s" % (n_features, alpha))
    plt.legend(loc="upper left")
    plt.xlabel("number of samples")
    plt.ylabel("Time (s)")
    plt.axis("tight")

    n_samples = 2000
    list_n_features = np.linspace(500, 3000, 5).astype(int)

    # Benchmarking with varying number of features and fixed number of samples
    lasso_results, lars_lasso_results = compute_bench(
        alpha, [n_samples], list_n_features, precompute=False
    )
    plt.subplot(212)
    plt.plot(list_n_features, lasso_results, "b-", label="Lasso")
    plt.plot(list_n_features, lars_lasso_results, "r-", label="LassoLars")
    plt.title("%d samples, alpha=%s" % (n_samples, alpha))
    plt.legend(loc="upper left")

    # Display the plot
    plt.show()
    # 设置 X 轴标签为 "number of features"
    # 设置 Y 轴标签为 "Time (s)"
    # 调整坐标轴范围以适应数据范围
    # 显示绘图窗口
    plt.xlabel("number of features")
    plt.ylabel("Time (s)")
    plt.axis("tight")
    plt.show()
```