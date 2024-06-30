# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\cluster.py`

```
# 从 sklearn 库中导入 KMeans 和 MiniBatchKMeans 类
from sklearn.cluster import KMeans, MiniBatchKMeans

# 导入 Benchmark, Estimator, Predictor, Transformer 类
from .common import Benchmark, Estimator, Predictor, Transformer
# 导入 _20newsgroups_highdim_dataset 和 _blobs_dataset 函数
from .datasets import _20newsgroups_highdim_dataset, _blobs_dataset
# 导入 neg_mean_inertia 函数
from .utils import neg_mean_inertia


# 创建 KMeansBenchmark 类，继承 Predictor, Transformer, Estimator, Benchmark 类
class KMeansBenchmark(Predictor, Transformer, Estimator, Benchmark):
    """
    Benchmarks for KMeans.
    """

    # 定义参数名列表
    param_names = ["representation", "algorithm", "init"]
    # 定义参数组合元组
    params = (["dense", "sparse"], ["lloyd", "elkan"], ["random", "k-means++"])

    # 设置缓存的方法
    def setup_cache(self):
        super().setup_cache()

    # 根据参数创建数据的方法
    def make_data(self, params):
        representation, algorithm, init = params

        # 根据 representation 参数选择数据集
        if representation == "sparse":
            data = _20newsgroups_highdim_dataset(n_samples=8000)
        else:
            data = _blobs_dataset(n_clusters=20)

        return data

    # 根据参数创建评估器的方法
    def make_estimator(self, params):
        representation, algorithm, init = params

        # 根据 representation 参数设置最大迭代次数
        max_iter = 30 if representation == "sparse" else 100

        # 创建 KMeans 对象并设置参数
        estimator = KMeans(
            n_clusters=20,
            algorithm=algorithm,
            init=init,
            n_init=1,
            max_iter=max_iter,
            tol=0,
            random_state=0,
        )

        return estimator

    # 创建评分器的方法
    def make_scorers(self):
        # 定义训练评分器的 lambda 函数
        self.train_scorer = lambda _, __: neg_mean_inertia(
            self.X, self.estimator.predict(self.X), self.estimator.cluster_centers_
        )
        # 定义测试评分器的 lambda 函数
        self.test_scorer = lambda _, __: neg_mean_inertia(
            self.X_val,
            self.estimator.predict(self.X_val),
            self.estimator.cluster_centers_,
        )


# 创建 MiniBatchKMeansBenchmark 类，继承 Predictor, Transformer, Estimator, Benchmark 类
class MiniBatchKMeansBenchmark(Predictor, Transformer, Estimator, Benchmark):
    """
    Benchmarks for MiniBatchKMeans.
    """

    # 定义参数名列表
    param_names = ["representation", "init"]
    # 定义参数组合元组
    params = (["dense", "sparse"], ["random", "k-means++"])

    # 设置缓存的方法
    def setup_cache(self):
        super().setup_cache()

    # 根据参数创建数据的方法
    def make_data(self, params):
        representation, init = params

        # 根据 representation 参数选择数据集
        if representation == "sparse":
            data = _20newsgroups_highdim_dataset()
        else:
            data = _blobs_dataset(n_clusters=20)

        return data

    # 根据参数创建评估器的方法
    def make_estimator(self, params):
        representation, init = params

        # 根据 representation 参数设置最大迭代次数
        max_iter = 5 if representation == "sparse" else 2

        # 创建 MiniBatchKMeans 对象并设置参数
        estimator = MiniBatchKMeans(
            n_clusters=20,
            init=init,
            n_init=1,
            max_iter=max_iter,
            batch_size=1000,
            max_no_improvement=None,
            compute_labels=False,
            random_state=0,
        )

        return estimator

    # 创建评分器的方法
    def make_scorers(self):
        # 定义训练评分器的 lambda 函数
        self.train_scorer = lambda _, __: neg_mean_inertia(
            self.X, self.estimator.predict(self.X), self.estimator.cluster_centers_
        )
        # 定义测试评分器的 lambda 函数
        self.test_scorer = lambda _, __: neg_mean_inertia(
            self.X_val,
            self.estimator.predict(self.X_val),
            self.estimator.cluster_centers_,
        )
```