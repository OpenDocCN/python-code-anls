# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\metrics.py`

```
# 从sklearn.metrics.pairwise模块导入pairwise_distances函数
from sklearn.metrics.pairwise import pairwise_distances

# 从当前目录下的common模块导入Benchmark类
from .common import Benchmark
# 从当前目录下的datasets模块导入_random_dataset函数
from .datasets import _random_dataset


# 定义PairwiseDistancesBenchmark类，继承Benchmark类，用于执行成对距离的基准测试
class PairwiseDistancesBenchmark(Benchmark):
    """
    Benchmarks for pairwise distances.
    """

    # 定义参数名称列表
    param_names = ["representation", "metric", "n_jobs"]
    # 定义参数组合，包括数据表示方式、距离度量方式和并行工作数
    params = (
        ["dense", "sparse"],  # 数据表示方式：密集和稀疏
        ["cosine", "euclidean", "manhattan", "correlation"],  # 距离度量方式
        Benchmark.n_jobs_vals,  # 并行工作数的取值
    )

    # 设置方法，根据给定的参数进行初始化设置
    def setup(self, *params):
        representation, metric, n_jobs = params

        # 如果数据表示方式为稀疏且距离度量方式为correlation，则抛出未实现错误
        if representation == "sparse" and metric == "correlation":
            raise NotImplementedError

        # 根据数据大小设置样本数
        if Benchmark.data_size == "large":
            if metric in ("manhattan", "correlation"):
                n_samples = 8000
            else:
                n_samples = 24000
        else:
            if metric in ("manhattan", "correlation"):
                n_samples = 4000
            else:
                n_samples = 12000

        # 根据样本数和数据表示方式生成随机数据集
        data = _random_dataset(n_samples=n_samples, representation=representation)
        # 将生成的数据集分配给类实例变量
        self.X, self.X_val, self.y, self.y_val = data

        # 设置成对距离计算的参数字典
        self.pdist_params = {"metric": metric, "n_jobs": n_jobs}

    # 定义方法用于测量成对距离计算的时间
    def time_pairwise_distances(self, *args):
        pairwise_distances(self.X, **self.pdist_params)

    # 定义方法用于测量成对距离计算的内存峰值
    def peakmem_pairwise_distances(self, *args):
        pairwise_distances(self.X, **self.pdist_params)
```