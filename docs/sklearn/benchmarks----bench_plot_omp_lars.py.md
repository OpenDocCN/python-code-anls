# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_omp_lars.py`

```
"""Benchmarks of orthogonal matching pursuit (:ref:`OMP`) versus least angle
regression (:ref:`least_angle_regression`)

The input data is mostly low rank but is a fat infinite tail.
"""

# 引入垃圾收集、系统模块和计时函数
import gc
import sys
from time import time

# 引入NumPy库
import numpy as np

# 引入sklearn库中的函数和类
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import lars_path, lars_path_gram, orthogonal_mp

# 定义一个函数用于计算基准测试结果
def compute_bench(samples_range, features_range):
    # 初始化迭代计数器
    it = 0

    # 初始化结果字典
    results = dict()

    # 创建空的NumPy数组，用于存储不同方法的结果
    lars = np.empty((len(features_range), len(samples_range)))
    lars_gram = lars.copy()
    omp = lars.copy()
    omp_gram = lars.copy()

    # 计算总的迭代次数
    max_it = len(samples_range) * len(features_range)
    # 遍历样本范围和特征范围
    for i_s, n_samples in enumerate(samples_range):
        for i_f, n_features in enumerate(features_range):
            # 迭代次数加一
            it += 1
            # 计算有信息量的特征数量
            n_informative = n_features // 10
            # 打印迭代信息
            print("====================")
            print("Iteration %03d of %03d" % (it, max_it))
            print("====================")
            # 设置数据集参数
            dataset_kwargs = {
                "n_samples": 1,
                "n_components": n_features,
                "n_features": n_samples,
                "n_nonzero_coefs": n_informative,
                "random_state": 0,
            }
            # 打印样本数量和特征数量
            print("n_samples: %d" % n_samples)
            print("n_features: %d" % n_features)
            # 生成稀疏编码信号
            y, X, _ = make_sparse_coded_signal(**dataset_kwargs)
            X = np.asfortranarray(X.T)

            # 垃圾回收
            gc.collect()
            # 打印信息，开始计时
            print("benchmarking lars_path (with Gram):", end="")
            sys.stdout.flush()
            tstart = time()
            # 计算 Gram 矩阵
            G = np.dot(X.T, X)  # precomputed Gram matrix
            Xy = np.dot(X.T, y)
            # 运行 lars_path 算法
            lars_path_gram(Xy=Xy, Gram=G, n_samples=y.size, max_iter=n_informative)
            delta = time() - tstart
            print("%0.3fs" % delta)
            lars_gram[i_f, i_s] = delta

            # 垃圾回收
            gc.collect()
            # 打印信息，开始计时
            print("benchmarking lars_path (without Gram):", end="")
            sys.stdout.flush()
            tstart = time()
            # 运行 lars_path 算法
            lars_path(X, y, Gram=None, max_iter=n_informative)
            delta = time() - tstart
            print("%0.3fs" % delta)
            lars[i_f, i_s] = delta

            # 垃圾回收
            gc.collect()
            # 打印信息，开始计时
            print("benchmarking orthogonal_mp (with Gram):", end="")
            sys.stdout.flush()
            tstart = time()
            # 运行 orthogonal_mp 算法
            orthogonal_mp(X, y, precompute=True, n_nonzero_coefs=n_informative)
            delta = time() - tstart
            print("%0.3fs" % delta)
            omp_gram[i_f, i_s] = delta

            # 垃圾回收
            gc.collect()
            # 打印信息，开始计时
            print("benchmarking orthogonal_mp (without Gram):", end="")
            sys.stdout.flush()
            tstart = time()
            # 运行 orthogonal_mp 算法
            orthogonal_mp(X, y, precompute=False, n_nonzero_coefs=n_informative)
            delta = time() - tstart
            print("%0.3fs" % delta)
            omp[i_f, i_s] = delta

    # 计算结果
    results["time(LARS) / time(OMP)\n (w/ Gram)"] = lars_gram / omp_gram
    results["time(LARS) / time(OMP)\n (w/o Gram)"] = lars / omp
    # 返回结果
    return results
# 如果这个脚本作为主程序被执行，则执行以下操作
if __name__ == "__main__":
    # 生成一个包含从1000到5000的5个整数的数组，作为样本范围
    samples_range = np.linspace(1000, 5000, 5).astype(int)
    # 生成一个包含从1000到5000的5个整数的数组，作为特征范围
    features_range = np.linspace(1000, 5000, 5).astype(int)
    # 调用 compute_bench 函数计算基准结果，返回一个字典
    results = compute_bench(samples_range, features_range)
    # 计算所有结果中的最大时间
    max_time = max(np.max(t) for t in results.values())

    # 导入 matplotlib.pyplot 库作为 plt
    import matplotlib.pyplot as plt

    # 创建一个名为 "scikit-learn OMP vs. LARS benchmark results" 的图像
    fig = plt.figure("scikit-learn OMP vs. LARS benchmark results")
    # 对结果字典中的每一项进行排序，并遍历它们
    for i, (label, timings) in enumerate(sorted(results.items())):
        # 在图像中添加子图，1行2列，当前为第 i+1 个子图
        ax = fig.add_subplot(1, 2, i + 1)
        # 计算颜色映射的上下限
        vmax = max(1 - timings.min(), -1 + timings.max())
        # 在子图上绘制矩阵图，使用颜色映射，不显示图像编号
        plt.matshow(timings, fignum=False, vmin=1 - vmax, vmax=1 + vmax)
        # 设置 x 轴刻度标签，包括空字符串和样本范围中的每个整数转换为字符串
        ax.set_xticklabels([""] + [str(each) for each in samples_range])
        # 设置 y 轴刻度标签，包括空字符串和特征范围中的每个整数转换为字符串
        ax.set_yticklabels([""] + [str(each) for each in features_range])
        # 设置 x 轴标签为 "n_samples"
        plt.xlabel("n_samples")
        # 设置 y 轴标签为 "n_features"
        plt.ylabel("n_features")
        # 设置子图标题为当前结果的标签
        plt.title(label)

    # 调整子图的布局参数，以便更好地适应图像大小
    plt.subplots_adjust(0.1, 0.08, 0.96, 0.98, 0.4, 0.63)
    # 在指定位置创建一个颜色条，水平方向
    ax = plt.axes([0.1, 0.08, 0.8, 0.06])
    plt.colorbar(cax=ax, orientation="horizontal")
    # 显示绘制的所有图像
    plt.show()
```