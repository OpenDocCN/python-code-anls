# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\decomposition.py`

```
# 导入需要的类和函数：PCA、DictionaryLearning、MiniBatchDictionaryLearning
from sklearn.decomposition import PCA, DictionaryLearning, MiniBatchDictionaryLearning

# 导入本地文件中的类和函数：Benchmark、Estimator、Transformer
from .common import Benchmark, Estimator, Transformer
from .datasets import _mnist_dataset, _olivetti_faces_dataset
from .utils import make_dict_learning_scorers, make_pca_scorers

# 定义一个名为PCABenchmark的类，继承自Transformer、Estimator、Benchmark
class PCABenchmark(Transformer, Estimator, Benchmark):
    """
    Benchmarks for PCA.
    """

    # 参数名列表
    param_names = ["svd_solver"]
    # 参数取值列表
    params = (["full", "arpack", "randomized"],)

    # 设置缓存的方法，调用父类的setup_cache方法
    def setup_cache(self):
        super().setup_cache()

    # 生成数据的方法，调用_mnist_dataset函数
    def make_data(self, params):
        return _mnist_dataset()

    # 生成评估器的方法，根据参数创建PCA对象并返回
    def make_estimator(self, params):
        (svd_solver,) = params
        estimator = PCA(n_components=32, svd_solver=svd_solver, random_state=0)
        return estimator

    # 生成评分器的方法，调用make_pca_scorers函数
    def make_scorers(self):
        make_pca_scorers(self)


# 定义一个名为DictionaryLearningBenchmark的类，继承自Transformer、Estimator、Benchmark
class DictionaryLearningBenchmark(Transformer, Estimator, Benchmark):
    """
    Benchmarks for DictionaryLearning.
    """

    # 参数名列表
    param_names = ["fit_algorithm", "n_jobs"]
    # 参数取值列表，其中n_jobs_vals来自Benchmark类的定义
    params = (["lars", "cd"], Benchmark.n_jobs_vals)

    # 设置缓存的方法，调用父类的setup_cache方法
    def setup_cache(self):
        super().setup_cache()

    # 生成数据的方法，调用_olivetti_faces_dataset函数
    def make_data(self, params):
        return _olivetti_faces_dataset()

    # 生成评估器的方法，根据参数创建DictionaryLearning对象并返回
    def make_estimator(self, params):
        fit_algorithm, n_jobs = params
        estimator = DictionaryLearning(
            n_components=15,
            fit_algorithm=fit_algorithm,
            alpha=0.1,
            transform_alpha=1,
            max_iter=20,
            tol=1e-16,
            random_state=0,
            n_jobs=n_jobs,
        )
        return estimator

    # 生成评分器的方法，调用make_dict_learning_scorers函数
    def make_scorers(self):
        make_dict_learning_scorers(self)


# 定义一个名为MiniBatchDictionaryLearningBenchmark的类，继承自Transformer、Estimator、Benchmark
class MiniBatchDictionaryLearningBenchmark(Transformer, Estimator, Benchmark):
    """
    Benchmarks for MiniBatchDictionaryLearning
    """

    # 参数名列表
    param_names = ["fit_algorithm", "n_jobs"]
    # 参数取值列表，其中n_jobs_vals来自Benchmark类的定义
    params = (["lars", "cd"], Benchmark.n_jobs_vals)

    # 设置缓存的方法，调用父类的setup_cache方法
    def setup_cache(self):
        super().setup_cache()

    # 生成数据的方法，调用_olivetti_faces_dataset函数
    def make_data(self, params):
        return _olivetti_faces_dataset()

    # 生成评估器的方法，根据参数创建MiniBatchDictionaryLearning对象并返回
    def make_estimator(self, params):
        fit_algorithm, n_jobs = params
        estimator = MiniBatchDictionaryLearning(
            n_components=15,
            fit_algorithm=fit_algorithm,
            alpha=0.1,
            batch_size=3,
            random_state=0,
            n_jobs=n_jobs,
        )
        return estimator

    # 生成评分器的方法，调用make_dict_learning_scorers函数
    def make_scorers(self):
        make_dict_learning_scorers(self)
```