# `D:\src\scipysrc\scikit-learn\benchmarks\bench_tsne_mnist.py`

```
"""
=============================
MNIST dataset T-SNE benchmark
=============================

"""

# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 数据
import os  # 提供操作系统相关的功能
import os.path as op  # 提供操作系统路径相关的功能
from time import time  # 导入时间函数

import numpy as np  # 数值计算库
from joblib import Memory  # 提供缓存功能

from sklearn.datasets import fetch_openml  # 从 OpenML 中获取数据集
from sklearn.decomposition import PCA  # PCA 主成分分析
from sklearn.manifold import TSNE  # t-SNE 降维算法
from sklearn.neighbors import NearestNeighbors  # 最近邻算法
from sklearn.utils import check_array  # 数据验证功能
from sklearn.utils import shuffle as _shuffle  # 数据洗牌函数
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads  # 多线程处理函数

LOG_DIR = "mnist_tsne_output"
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

memory = Memory(os.path.join(LOG_DIR, "mnist_tsne_benchmark_data"), mmap_mode="r")


@memory.cache
def load_data(dtype=np.float32, order="C", shuffle=True, seed=0):
    """Load the data, then cache and memmap the train/test split"""
    print("Loading dataset...")
    data = fetch_openml("mnist_784", as_frame=True)

    X = check_array(data["data"], dtype=dtype, order=order)
    y = data["target"]

    if shuffle:
        X, y = _shuffle(X, y, random_state=seed)

    # Normalize features
    X /= 255
    return X, y


def nn_accuracy(X, X_embedded, k=1):
    """Accuracy of the first nearest neighbor"""
    knn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    _, neighbors_X = knn.fit(X).kneighbors()
    _, neighbors_X_embedded = knn.fit(X_embedded).kneighbors()
    return np.mean(neighbors_X == neighbors_X_embedded)


def tsne_fit_transform(model, data):
    """Fit t-SNE model and transform data"""
    transformed = model.fit_transform(data)
    return transformed, model.n_iter_


def sanitize(filename):
    """Sanitize filename for safe usage"""
    return filename.replace("/", "-").replace(" ", "_")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark for t-SNE")
    parser.add_argument(
        "--order", type=str, default="C", help="Order of the input data"
    )
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument(
        "--bhtsne",
        action="store_true",
        help=(
            "if set and the reference bhtsne code is "
            "correctly installed, run it in the benchmark."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "if set, run the benchmark with the whole MNIST."
            "dataset. Note that it will take up to 1 hour."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="if set, run the benchmark with a memory profiler.",
    )
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of principal components for preprocessing.",
    )
    args = parser.parse_args()

    print("Used number of threads: {}".format(_openmp_effective_n_threads()))
    X, y = load_data(order=args.order)
    # 如果设置了 PCA 组件数量大于 0
    if args.pca_components > 0:
        # 记录开始时间
        t0 = time()
        # 使用 PCA 进行降维，并转换数据 X
        X = PCA(n_components=args.pca_components).fit_transform(X)
        # 打印 PCA 预处理降至特定维度所花费的时间
        print(
            "PCA preprocessing down to {} dimensions took {:0.3f}s".format(
                args.pca_components, time() - t0
            )
        )

    # 创建一个空的方法列表
    methods = []

    # 将 TSNE 实例化并添加到 methods 列表中
    tsne = TSNE(
        n_components=2,              # 降维后的维度设为 2
        init="pca",                  # 使用 PCA 进行初始化
        perplexity=args.perplexity,  # 设定 perplexity 参数
        verbose=args.verbose,        # 设定是否输出详细信息
        n_iter=1000,                 # 设定迭代次数
    )
    methods.append(("sklearn TSNE", lambda data: tsne_fit_transform(tsne, data)))

    # 如果启用了 bhtsne
    if args.bhtsne:
        try:
            # 尝试导入 bhtsne 库中的 run_bh_tsne 函数
            from bhtsne.bhtsne import run_bh_tsne
        except ImportError as e:
            # 若导入失败，抛出 ImportError 异常
            raise ImportError(
                """
    """
    If you want comparison with the reference implementation, build the
    binary from source (https://github.com/lvdmaaten/bhtsne) in the folder
    benchmarks/bhtsne and add an empty `__init__.py` file in the folder:

    $ git clone git@github.com:lvdmaaten/bhtsne.git
    $ cd bhtsne
    $ g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2
    $ touch __init__.py
    $ cd ..
    """
    ) from e

    def bhtsne(X):
        """Wrapper for the reference lvdmaaten/bhtsne implementation."""
        # PCA preprocessing is done elsewhere in the benchmark script
        n_iter = -1  # TODO find a way to report the number of iterations
        return (
            run_bh_tsne(
                X,
                use_pca=False,
                perplexity=args.perplexity,
                verbose=args.verbose > 0,
            ),
            n_iter,
        )

    methods.append(("lvdmaaten/bhtsne", bhtsne))

if args.profile:
    try:
        from memory_profiler import profile
    except ImportError as e:
        # Raise ImportError with instructions to install memory_profiler
        raise ImportError(
            "To run the benchmark with `--profile`, you "
            "need to install `memory_profiler`. Please "
            "run `pip install memory_profiler`."
        ) from e
    # Profile each method in methods list
    methods = [(n, profile(m)) for n, m in methods]

# Define different data sizes for benchmarking
data_size = [100, 500, 1000, 5000, 10000]
if args.all:
    data_size.append(70000)

# Initialize an empty list to store benchmarking results
results = []
# Get base filename without extension
basename = os.path.basename(os.path.splitext(__file__)[0])
# Define the log file path
log_filename = os.path.join(LOG_DIR, basename + ".json")

# Iterate over different data sizes
for n in data_size:
    # Select subsets of data for training
    X_train = X[:n]
    y_train = y[:n]
    n = X_train.shape[0]  # Update n to current subset size
    # Iterate over each method in the methods list
    for name, method in methods:
        # Print a message indicating the current fitting process
        print("Fitting {} on {} samples...".format(name, n))
        t0 = time()  # Record start time
        # Save original data arrays as numpy files
        np.save(
            os.path.join(LOG_DIR, "mnist_{}_{}.npy".format("original", n)), X_train
        )
        np.save(
            os.path.join(LOG_DIR, "mnist_{}_{}.npy".format("original_labels", n)),
            y_train,
        )
        # Apply the embedding method to X_train
        X_embedded, n_iter = method(X_train)
        duration = time() - t0  # Calculate elapsed time
        # Calculate nearest neighbor accuracy
        precision_5 = nn_accuracy(X_train, X_embedded)
        # Print fitting summary including time taken and accuracy
        print(
            "Fitting {} on {} samples took {:.3f}s in {:d} iterations, "
            "nn accuracy: {:0.3f}".format(name, n, duration, n_iter, precision_5)
        )
        # Append benchmarking results to list
        results.append(dict(method=name, duration=duration, n_samples=n))
        # Write results to a JSON log file
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(results, f)
        # Sanitize method name for file naming
        method_name = sanitize(name)
        # Save embedded data as numpy file
        np.save(
            op.join(LOG_DIR, "mnist_{}_{}.npy".format(method_name, n)), X_embedded
        )
```