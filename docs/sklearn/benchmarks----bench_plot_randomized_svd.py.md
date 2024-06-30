# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_randomized_svd.py`

```
# Benchmarks on the power iterations phase in randomized SVD.
#
# We test on various synthetic and real datasets the effect of increasing
# the number of power iterations in terms of quality of approximation
# and running time. A number greater than 0 should help with noisy matrices,
# which are characterized by a slow spectral decay.
#
# We test several policies for normalizing the power iterations. Normalization
# is crucial to avoid numerical issues.
#
# The quality of the approximation is measured by the spectral norm discrepancy
# between the original input matrix and the reconstructed one (by multiplying
# the randomized_svd's outputs). The spectral norm is always equivalent to the
# largest singular value of a matrix. (3) justifies this choice. However, one can
# notice in these experiments that Frobenius and spectral norms behave
# very similarly in a qualitative sense. Therefore, we suggest to run these
# benchmarks with `enable_spectral_norm = False`, as Frobenius' is MUCH faster to
# compute.
#
# The benchmarks follow.
#
# (a) plot: time vs norm, varying number of power iterations
#     data: many datasets
#     goal: compare normalization policies and study how the number of power
#     iterations affect time and norm
#
# (b) plot: n_iter vs norm, varying rank of data and number of components for
#     randomized_SVD
#     data: low-rank matrices on which we control the rank
#     goal: study whether the rank of the matrix and the number of components
#     extracted by randomized SVD affect "the optimal" number of power iterations
#
# (c) plot: time vs norm, varying datasets
#     data: many datasets
#     goal: compare default configurations
#
# We compare the following algorithms:
# - randomized_svd(..., power_iteration_normalizer='none')
# - randomized_svd(..., power_iteration_normalizer='LU')
# - randomized_svd(..., power_iteration_normalizer='QR')
# - randomized_svd(..., power_iteration_normalizer='auto')
# - fbpca.pca() from https://github.com/facebook/fbpca (if installed)
#
# Conclusion
# ----------
# - n_iter=2 appears to be a good default value
# - power_iteration_normalizer='none' is OK if n_iter is small, otherwise LU
#   gives similar errors to QR but is cheaper. That's what 'auto' implements.
#
# References
# ----------
# (1) :arxiv:`"Finding structure with randomness:
#     Stochastic algorithms for constructing approximate matrix decompositions."
#     <0909.4061>`
#     Halko, et al., (2009)
#
# (2) A randomized algorithm for the decomposition of matrices
#     Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
#
# (3) An implementation of a randomized algorithm for principal component
#     analysis
#     A. Szlam et al. 2014

# Author: Giorgio Patrini

import gc                    # Importing garbage collector module for memory management
import os.path               # Importing module for common pathname manipulations
import pickle                # Importing module to serialize and deserialize Python objects
from collections import defaultdict  # Importing defaultdict for default dictionary values
from time import time        # Importing time function from time module

import matplotlib.pyplot as plt  # Importing pyplot module from matplotlib for plotting
import numpy as np           # Importing numpy for numerical computations
import scipy as sp           # Importing scipy for scientific and technical computing

from sklearn.datasets import (  # Importing specific functions from sklearn.datasets
    fetch_20newsgroups_vectorized,
    fetch_lfw_people,
    fetch_olivetti_faces,
    fetch_openml,
    fetch_rcv1,
    make_low_rank_matrix,
    make_sparse_uncorrelated,


    # 导入函数 make_sparse_uncorrelated
    # 这行代码导入了名为 make_sparse_uncorrelated 的函数，可以在后续的代码中使用这个函数来生成稀疏且不相关的数据。
)
# 从 sklearn.utils 导入 gen_batches 函数
from sklearn.utils import gen_batches
# 从 sklearn.utils._arpack 导入 _init_arpack_v0 函数
from sklearn.utils._arpack import _init_arpack_v0
# 从 sklearn.utils.extmath 导入 randomized_svd 函数
from sklearn.utils.extmath import randomized_svd
# 从 sklearn.utils.validation 导入 check_random_state 函数
from sklearn.utils.validation import check_random_state

try:
    # 尝试导入 fbpca 库
    import fbpca
    # 若导入成功，则设置 fbpca_available 为 True
    fbpca_available = True
except ImportError:
    # 若导入失败，则设置 fbpca_available 为 False
    fbpca_available = False

# 如果启用了此选项，则测试速度较慢，并且在大数据下可能会崩溃
enable_spectral_norm = False

# TODO: 使用幂法计算近似谱范数，参考 Estimating the largest eigenvalues by the power and Lanczos methods with a random start, Jacek Kuczynski and Henryk Wozniakowski, SIAM Journal on Matrix Analysis and Applications, 13 (4): 1094-1122, 1992.
# 这种近似是谱范数的一个非常快速的估计，但依赖于起始的随机向量。

# 决定何时切换到批处理计算矩阵范数，以防重构（密集）矩阵过大
MAX_MEMORY = int(4e9)

# 下列数据集可以从以下链接手动下载：
# CIFAR 10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# SVHN: http://ufldl.stanford.edu/housenumbers/train_32x32.mat
CIFAR_FOLDER = "./cifar-10-batches-py/"
SVHN_FOLDER = "./SVHN/"

# 定义数据集列表
datasets = [
    "low rank matrix",
    "lfw_people",
    "olivetti_faces",
    "20newsgroups",
    "mnist_784",
    "CIFAR",
    "a3a",
    "SVHN",
    "uncorrelated matrix",
]

# 定义大稀疏数据集列表
big_sparse_datasets = ["big sparse matrix", "rcv1"]


def unpickle(file_name):
    # 打开文件并加载 pickle 格式的数据
    with open(file_name, "rb") as fo:
        return pickle.load(fo, encoding="latin1")["data"]


def handle_missing_dataset(file_folder):
    # 如果文件夹不存在，则打印提示信息并跳过测试
    if not os.path.isdir(file_folder):
        print("%s file folder not found. Test skipped." % file_folder)
        return 0


def get_data(dataset_name):
    # 打印正在获取数据集的信息
    print("Getting dataset: %s" % dataset_name)

    if dataset_name == "lfw_people":
        # 获取 LFW 人脸数据集的数据
        X = fetch_lfw_people().data
    elif dataset_name == "20newsgroups":
        # 获取 20newsgroups 数据集的部分数据
        X = fetch_20newsgroups_vectorized().data[:, :100000]
    elif dataset_name == "olivetti_faces":
        # 获取 Olivetti 人脸数据集的数据
        X = fetch_olivetti_faces().data
    elif dataset_name == "rcv1":
        # 获取 RCV1 数据集的数据
        X = fetch_rcv1().data
    elif dataset_name == "CIFAR":
        # 如果 CIFAR 数据集文件夹不存在，则返回
        if handle_missing_dataset(CIFAR_FOLDER) == 0:
            return
        # 读取 CIFAR 数据集的各批次数据并堆叠起来
        X1 = [unpickle("%sdata_batch_%d" % (CIFAR_FOLDER, i + 1)) for i in range(5)]
        X = np.vstack(X1)
        del X1
    elif dataset_name == "SVHN":
        # 如果 SVHN 数据集文件夹不存在，则返回
        if handle_missing_dataset(SVHN_FOLDER) == 0:
            return
        # 加载 SVHN 数据集的数据
        X1 = sp.io.loadmat("%strain_32x32.mat" % SVHN_FOLDER)["X"]
        X2 = [X1[:, :, :, i].reshape(32 * 32 * 3) for i in range(X1.shape[3])]
        X = np.vstack(X2)
        del X1
        del X2
    elif dataset_name == "low rank matrix":
        # 创建低秩矩阵数据
        X = make_low_rank_matrix(
            n_samples=500,
            n_features=int(1e4),
            effective_rank=100,
            tail_strength=0.5,
            random_state=random_state,
        )
    # 如果数据集名称为 "uncorrelated matrix"，生成一个稀疏且不相关的矩阵
    elif dataset_name == "uncorrelated matrix":
        # 使用 make_sparse_uncorrelated 函数生成稀疏矩阵 X
        X, _ = make_sparse_uncorrelated(
            n_samples=500, n_features=10000, random_state=random_state
        )
    
    # 如果数据集名称为 "big sparse matrix"，生成一个大型稀疏矩阵
    elif dataset_name == "big sparse matrix":
        # 设置稀疏度和矩阵大小
        sparsity = int(1e6)
        size = int(1e6)
        small_size = int(1e4)
        
        # 生成正态分布的数据
        data = np.random.normal(0, 1, int(sparsity / 10))
        data = np.repeat(data, 10)
        
        # 生成随机的行和列索引
        row = np.random.uniform(0, small_size, sparsity)
        col = np.random.uniform(0, small_size, sparsity)
        
        # 创建稀疏矩阵 X，使用 CSR 格式存储
        X = sp.sparse.csr_matrix((data, (row, col)), shape=(size, small_size))
        
        # 释放临时使用的变量内存
        del data
        del row
        del col
    
    # 如果数据集名称不是上述两种情况，从 OpenML 中获取数据集 X 的数据部分
    else:
        X = fetch_openml(dataset_name).data
    
    # 返回生成或获取的数据集 X
    return X
# 绘制时间与归一化差异的折线图
def plot_time_vs_s(time, norm, point_labels, title):
    # 创建一个新的图形
    plt.figure()
    
    # 颜色列表
    colors = ["g", "b", "y"]
    
    # 对归一化字典按键进行排序并枚举处理
    for i, l in enumerate(sorted(norm.keys())):
        # 如果键不是"fbpca"
        if l != "fbpca":
            # 绘制折线图，并标记数据点和颜色
            plt.plot(time[l], norm[l], label=l, marker="o", c=colors.pop())
        else:
            # 对于"fbpca"键，绘制折线图，并标记数据点和红色标记
            plt.plot(time[l], norm[l], label=l, marker="^", c="red")

        # 添加数据点的标签注释
        for label, x, y in zip(point_labels, list(time[l]), list(norm[l])):
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(0, -20),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )
    
    # 添加图例到右上角
    plt.legend(loc="upper right")
    
    # 设置图形标题
    plt.suptitle(title)
    
    # 设置y轴标签
    plt.ylabel("norm discrepancy")
    
    # 设置x轴标签
    plt.xlabel("running time [s]")


# 绘制时间与归一化差异的散点图
def scatter_time_vs_s(time, norm, point_labels, title):
    # 创建一个新的图形
    plt.figure()
    
    # 散点的默认大小
    size = 100
    
    # 对归一化字典按键进行排序并枚举处理
    for i, l in enumerate(sorted(norm.keys())):
        # 如果键不是"fbpca"
        if l != "fbpca":
            # 绘制散点图，并标记数据点、蓝色标记和指定大小
            plt.scatter(time[l], norm[l], label=l, marker="o", c="b", s=size)
            # 添加数据点的标签注释
            for label, x, y in zip(point_labels, list(time[l]), list(norm[l])):
                plt.annotate(
                    label,
                    xy=(x, y),
                    xytext=(0, -80),
                    textcoords="offset points",
                    ha="right",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                    va="bottom",
                    size=11,
                    rotation=90,
                )
        else:
            # 对于"fbpca"键，绘制散点图，并标记数据点、红色标记和指定大小
            plt.scatter(time[l], norm[l], label=l, marker="^", c="red", s=size)
            # 添加数据点的标签注释
            for label, x, y in zip(point_labels, list(time[l]), list(norm[l])):
                plt.annotate(
                    label,
                    xy=(x, y),
                    xytext=(0, 30),
                    textcoords="offset points",
                    ha="right",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                    va="bottom",
                    size=11,
                    rotation=90,
                )

    # 添加图例，自动选择最佳位置
    plt.legend(loc="best")
    
    # 设置图形标题
    plt.suptitle(title)
    
    # 设置y轴标签
    plt.ylabel("norm discrepancy")
    
    # 设置x轴标签
    plt.xlabel("running time [s]")


# 绘制幂迭代次数与归一化差异的折线图
def plot_power_iter_vs_s(power_iter, s, title):
    # 创建一个新的图形
    plt.figure()
    
    # 对归一化字典按键进行排序并处理
    for l in sorted(s.keys()):
        # 绘制折线图，并标记数据点为圆圈形状
        plt.plot(power_iter, s[l], label=l, marker="o")
    
    # 添加图例到右下角，设置图例字体大小为10
    plt.legend(loc="lower right", prop={"size": 10})
    
    # 设置图形标题
    plt.suptitle(title)
    
    # 设置y轴标签
    plt.ylabel("norm discrepancy")
    
    # 设置x轴标签
    plt.xlabel("n_iter")


def svd_timing(
    X, n_comps, n_iter, n_oversamples, power_iteration_normalizer="auto", method=None
):
    """
    Measure time for decomposition
    """
    # 打印消息，表示正在运行SVD分解
    print("... running SVD ...")
    # 如果方法不是 "fbpca"，执行以下代码块
    if method != "fbpca":
        # 手动进行垃圾回收
        gc.collect()
        # 记录当前时间
        t0 = time()
        # 执行随机化SVD分解
        U, mu, V = randomized_svd(
            X,  # 要分解的数据矩阵
            n_comps,  # 分解后的主成分个数
            n_oversamples=n_oversamples,  # 随机采样时额外采样的数量
            n_iter=n_iter,  # 迭代次数
            power_iteration_normalizer=power_iteration_normalizer,  # 幂迭代的标准化方法
            random_state=random_state,  # 随机数生成器的种子
            transpose=False,  # 是否转置输入矩阵
        )
        # 计算函数调用所花费的时间
        call_time = time() - t0
    # 如果方法是 "fbpca"，执行以下代码块
    else:
        # 手动进行垃圾回收
        gc.collect()
        # 记录当前时间
        t0 = time()
        # 执行另一种PCA方法（fbpca库中的实现）
        # 注意：这里的l参数的约定可能与标准不同
        U, mu, V = fbpca.pca(
            X,  # 要分解的数据矩阵
            n_comps,  # 分解后的主成分个数
            raw=True,  # 是否返回原始的PCA结果
            n_iter=n_iter,  # 迭代次数
            l=n_oversamples + n_comps  # fbpca中的一个参数，通常与n_comps相关
        )
        # 计算函数调用所花费的时间
        call_time = time() - t0

    # 返回分解得到的结果U、mu、V以及函数调用的时间
    return U, mu, V, call_time
# 计算矩阵与原始矩阵之间的规范化差异，当随机化SVD使用给定参数调用时

def norm_diff(A, norm=2, msg=True, random_state=None):
    """
    Compute the norm diff with the original matrix, when randomized
    SVD is called with *params.

    norm: 2 => spectral; 'fro' => Frobenius
    """

    if msg:
        # 如果msg为True，则打印计算的规范化差异类型
        print("... computing %s norm ..." % norm)
    if norm == 2:
        # 如果norm为2，进行随机化SVD计算
        v0 = _init_arpack_v0(min(A.shape), random_state)
        value = sp.sparse.linalg.svds(A, k=1, return_singular_vectors=False, v0=v0)
    else:
        # 如果norm不为2，则根据A的类型计算相应的规范化差异
        if sp.sparse.issparse(A):
            value = sp.sparse.linalg.norm(A, ord=norm)
        else:
            value = sp.linalg.norm(A, ord=norm)
    return value


def scalable_frobenius_norm_discrepancy(X, U, s, V):
    # 如果X不是稀疏矩阵，或者稀疏但大小不超过MAX_MEMORY限制
    if not sp.sparse.issparse(X) or (
        X.shape[0] * X.shape[1] * X.dtype.itemsize < MAX_MEMORY
    ):
        # X - U.dot(np.diag(s).dot(V)) 将适合内存，计算Frobenius范数的差异
        A = X - U.dot(np.diag(s).dot(V))
        return norm_diff(A, norm="fro")

    # 如果X是稀疏矩阵且大小超过MAX_MEMORY限制，以批处理方式计算Frobenius范数的差异
    print("... computing fro norm by batches...")
    batch_size = 1000
    Vhat = np.diag(s).dot(V)
    cum_norm = 0.0
    for batch in gen_batches(X.shape[0], batch_size):
        # 计算每个批次中的差异，并累积计算结果
        M = X[batch, :] - U[batch, :].dot(Vhat)
        cum_norm += norm_diff(M, norm="fro", msg=False)
    return np.sqrt(cum_norm)


def bench_a(X, dataset_name, power_iter, n_oversamples, n_comps):
    # 初始化默认字典以记录不同类型的计时和结果
    all_time = defaultdict(list)
    if enable_spectral_norm:
        # 如果启用了谱范数，初始化另一个默认字典以记录谱范数相关的结果
        all_spectral = defaultdict(list)
        # 计算X的谱范数的规范化差异
        X_spectral_norm = norm_diff(X, norm=2, msg=False, random_state=0)
    # 初始化记录Frobenius范数相关结果的默认字典
    all_frobenius = defaultdict(list)
    # 计算X的Frobenius范数的规范化差异
    X_fro_norm = norm_diff(X, norm="fro", msg=False)
    # 对于每个给定的 power_iter 值进行迭代
    for pi in power_iter:
        # 对于每种指定的 power iteration normalizer 方法进行迭代
        for pm in ["none", "LU", "QR"]:
            # 打印当前 sklearn 方法的迭代次数和 normalizer 方法
            print("n_iter = %d on sklearn - %s" % (pi, pm))
            # 执行 SVD 计时，并返回 U, s, V 以及运行时间
            U, s, V, time = svd_timing(
                X,
                n_comps,
                n_iter=pi,
                power_iteration_normalizer=pm,
                n_oversamples=n_oversamples,
            )
            # 生成用于标记的字符串，表示使用的 sklearn 方法和 normalizer
            label = "sklearn - %s" % pm
            # 将当前方法的运行时间记录到 all_time 字典中
            all_time[label].append(time)
            # 如果启用了 spectral norm 的计算，则计算当前重构矩阵 A 的 spectral norm 差异
            if enable_spectral_norm:
                A = U.dot(np.diag(s).dot(V))
                all_spectral[label].append(
                    norm_diff(X - A, norm=2, random_state=0) / X_spectral_norm
                )
            # 计算当前重构矩阵与原始矩阵之间的 Frobenius norm 差异，并记录到 all_frobenius 字典中
            f = scalable_frobenius_norm_discrepancy(X, U, s, V)
            all_frobenius[label].append(f / X_fro_norm)

        # 如果可用 fbpca 方法
        if fbpca_available:
            # 打印当前 fbpca 方法的迭代次数
            print("n_iter = %d on fbca" % (pi))
            # 执行 SVD 计时，使用 fbpca 方法，并返回 U, s, V 以及运行时间
            U, s, V, time = svd_timing(
                X,
                n_comps,
                n_iter=pi,
                power_iteration_normalizer=pm,
                n_oversamples=n_oversamples,
                method="fbpca",
            )
            # 设置用于标记的字符串，表示使用的 fbpca 方法
            label = "fbpca"
            # 将当前方法的运行时间记录到 all_time 字典中
            all_time[label].append(time)
            # 如果启用了 spectral norm 的计算，则计算当前重构矩阵 A 的 spectral norm 差异
            if enable_spectral_norm:
                A = U.dot(np.diag(s).dot(V))
                all_spectral[label].append(
                    norm_diff(X - A, norm=2, random_state=0) / X_spectral_norm
                )
            # 计算当前重构矩阵与原始矩阵之间的 Frobenius norm 差异，并记录到 all_frobenius 字典中
            f = scalable_frobenius_norm_discrepancy(X, U, s, V)
            all_frobenius[label].append(f / X_fro_norm)

    # 如果启用 spectral norm 的计算，则生成标题并绘制时间与 spectral norm 差异的图表
    if enable_spectral_norm:
        title = "%s: spectral norm diff vs running time" % (dataset_name)
        plot_time_vs_s(all_time, all_spectral, power_iter, title)
    
    # 生成标题并绘制时间与 Frobenius norm 差异的图表
    title = "%s: Frobenius norm diff vs running time" % (dataset_name)
    plot_time_vs_s(all_time, all_frobenius, power_iter, title)
# 定义函数 bench_b，用于评估在给定数据集上执行低秩矩阵近似的性能
def bench_b(power_list):
    # 设置数据集参数：样本数为 1000，特征数为 10000
    n_samples, n_features = 1000, 10000
    # 定义数据集参数字典
    data_params = {
        "n_samples": n_samples,
        "n_features": n_features,
        "tail_strength": 0.7,
        "random_state": random_state,
    }
    # 创建数据集名称字符串
    dataset_name = "low rank matrix %d x %d" % (n_samples, n_features)
    # 定义矩阵的秩列表
    ranks = [10, 50, 100]

    # 如果启用谱范数计算，则创建谱范数结果的默认字典
    if enable_spectral_norm:
        all_spectral = defaultdict(list)
    # 创建Frobenius范数结果的默认字典
    all_frobenius = defaultdict(list)

    # 遍历不同的秩
    for rank in ranks:
        # 生成指定秩的低秩矩阵 X
        X = make_low_rank_matrix(effective_rank=rank, **data_params)
        # 如果启用谱范数计算，计算谱范数差异
        if enable_spectral_norm:
            X_spectral_norm = norm_diff(X, norm=2, msg=False, random_state=0)
        # 计算Frobenius范数差异
        X_fro_norm = norm_diff(X, norm="fro", msg=False)

        # 遍历不同的 n_comp 值
        for n_comp in [int(rank / 2), rank, rank * 2]:
            # 创建当前参数组合的标签
            label = "rank=%d, n_comp=%d" % (rank, n_comp)
            # 打印标签信息
            print(label)
            # 遍历给定的迭代次数列表
            for pi in power_list:
                # 执行SVD计时，返回U、s、V等信息
                U, s, V, _ = svd_timing(
                    X,
                    n_comp,
                    n_iter=pi,
                    n_oversamples=2,
                    power_iteration_normalizer="LU",
                )
                # 如果启用谱范数计算，计算谱范数差异并存储
                if enable_spectral_norm:
                    A = U.dot(np.diag(s).dot(V))
                    all_spectral[label].append(
                        norm_diff(X - A, norm=2, random_state=0) / X_spectral_norm
                    )
                # 计算并存储可扩展Frobenius范数差异
                f = scalable_frobenius_norm_discrepancy(X, U, s, V)
                all_frobenius[label].append(f / X_fro_norm)

    # 如果启用谱范数计算，创建谱范数结果绘图的标题
    if enable_spectral_norm:
        title = "%s: spectral norm diff vs n power iteration" % (dataset_name)
        # 绘制谱范数差异随迭代次数变化的图表
        plot_power_iter_vs_s(power_iter, all_spectral, title)
    # 创建Frobenius范数结果绘图的标题
    title = "%s: Frobenius norm diff vs n power iteration" % (dataset_name)
    # 绘制Frobenius范数差异随迭代次数变化的图表
    plot_power_iter_vs_s(power_iter, all_frobenius, title)
    # 对每个数据集名称进行迭代
    for dataset_name in datasets:
        # 获取数据集 X
        X = get_data(dataset_name)
        # 如果数据集 X 为空，则跳过当前循环，继续下一个数据集
        if X is None:
            continue

        # 如果启用了谱范数归一化
        if enable_spectral_norm:
            # 计算 X 的谱范数差异，并将结果存储在 X_spectral_norm 中
            X_spectral_norm = norm_diff(X, norm=2, msg=False, random_state=0)
        
        # 计算 X 的弗罗贝尼乌斯范数差异，并将结果存储在 X_fro_norm 中
        X_fro_norm = norm_diff(X, norm="fro", msg=False)
        
        # 计算 n_comps 和 X.shape 的最小值，并赋值给 n_comps
        n_comps = np.minimum(n_comps, np.min(X.shape))

        # 设置 label 为 "sklearn"，打印数据集的信息
        label = "sklearn"
        print("%s %d x %d - %s" % (dataset_name, X.shape[0], X.shape[1], label))
        
        # 执行奇异值分解，并记录执行时间
        U, s, V, time = svd_timing(X, n_comps, n_iter=2, n_oversamples=10, method=label)

        # 将执行时间 time 添加到 all_time[label] 列表中
        all_time[label].append(time)
        
        # 如果启用了谱范数归一化
        if enable_spectral_norm:
            # 重构矩阵 A，并计算重构误差的归一化谱范数差异，并将结果添加到 all_spectral[label] 列表中
            A = U.dot(np.diag(s).dot(V))
            all_spectral[label].append(
                norm_diff(X - A, norm=2, random_state=0) / X_spectral_norm
            )
        
        # 计算可扩展弗罗贝尼乌斯范数差异，并将结果添加到 all_frobenius[label] 列表中
        f = scalable_frobenius_norm_discrepancy(X, U, s, V)
        all_frobenius[label].append(f / X_fro_norm)

        # 如果 fbpca 可用
        if fbpca_available:
            # 设置 label 为 "fbpca"，打印数据集的信息
            label = "fbpca"
            print("%s %d x %d - %s" % (dataset_name, X.shape[0], X.shape[1], label))
            
            # 执行奇异值分解，并记录执行时间
            U, s, V, time = svd_timing(
                X, n_comps, n_iter=2, n_oversamples=2, method=label
            )
            
            # 将执行时间 time 添加到 all_time[label] 列表中
            all_time[label].append(time)
            
            # 如果启用了谱范数归一化
            if enable_spectral_norm:
                # 重构矩阵 A，并计算重构误差的归一化谱范数差异，并将结果添加到 all_spectral[label] 列表中
                A = U.dot(np.diag(s).dot(V))
                all_spectral[label].append(
                    norm_diff(X - A, norm=2, random_state=0) / X_spectral_norm
                )
            
            # 计算可扩展弗罗贝尼乌斯范数差异，并将结果添加到 all_frobenius[label] 列表中
            f = scalable_frobenius_norm_discrepancy(X, U, s, V)
            all_frobenius[label].append(f / X_fro_norm)

    # 如果 all_time 字典长度为0，抛出 ValueError 异常
    if len(all_time) == 0:
        raise ValueError("No tests ran. Aborting.")

    # 如果启用了谱范数归一化
    if enable_spectral_norm:
        # 设置标题为 "normalized spectral norm diff vs running time"，绘制散点图
        title = "normalized spectral norm diff vs running time"
        scatter_time_vs_s(all_time, all_spectral, datasets, title)
    
    # 设置标题为 "normalized Frobenius norm diff vs running time"，绘制散点图
    title = "normalized Frobenius norm diff vs running time"
    scatter_time_vs_s(all_time, all_frobenius, datasets, title)
# 如果这个脚本被直接运行（而不是被作为模块导入），则执行以下代码块
if __name__ == "__main__":
    # 使用种子1234初始化随机数生成器的状态
    random_state = check_random_state(1234)

    # 生成一个包含0到5的整数数组，表示幂迭代次数
    power_iter = np.arange(0, 6)
    
    # 设置主成分分析（PCA）的组件数目为50
    n_comps = 50

    # 遍历数据集列表中的每个数据集名称
    for dataset_name in datasets:
        # 从数据集名称获取数据集X
        X = get_data(dataset_name)
        
        # 如果数据集X为空，则继续下一个循环
        if X is None:
            continue
        
        # 打印当前数据集的信息，包括数据集名称、数据集的行数和列数
        print(
            " >>>>>> Benching sklearn and fbpca on %s %d x %d"
            % (dataset_name, X.shape[0], X.shape[1])
        )
        
        # 执行性能基准测试A，传入数据集X、数据集名称、幂迭代次数、过采样次数、主成分数目
        bench_a(
            X,
            dataset_name,
            power_iter,
            n_oversamples=2,
            n_comps=np.minimum(n_comps, np.min(X.shape)),
        )

    # 打印信息，表明接下来将在模拟的低秩矩阵上进行基准测试，秩数可变
    print(" >>>>>> Benching on simulated low rank matrix with variable rank")
    
    # 执行性能基准测试B，传入幂迭代次数数组
    bench_b(power_iter)

    # 打印信息，表明接下来将在sklearn和fbpca的默认配置上进行基准测试
    print(" >>>>>> Benching sklearn and fbpca default configurations")
    
    # 执行性能基准测试C，传入数据集列表和大稀疏数据集列表以及主成分数目
    bench_c(datasets + big_sparse_datasets, n_comps)

    # 显示绘图结果
    plt.show()
```